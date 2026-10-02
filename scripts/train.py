"""configs/train.yaml로 YOLO26s 파인튜닝 (README "학습").

학습 중에는 Ultralytics의 배치 진행 막대 아래에, 에폭이 끝날 때마다 현황판을 출력한다.
전체 진행률, 학습률, 손실, 검증 지표, 최고 기록, 얼리 스톱까지 남은 에폭을 보여준다.

중단 후 재개:
- Ultralytics가 에폭마다 weights/last.pt(옵티마이저 포함)를 저장한다. 저장 직후 last_backup.pt로 한 벌 더
  복사한다 (임시 파일에 쓰고 이름을 바꾸므로 복사 중 전원이 꺼져도 이전 백업이 남는다).
- 다시 실행하면 runs/<name>* 중 가장 최근 학습이 끝나지 않았을 때 마지막 에폭 다음부터 이어서 학습한다.
  last.pt가 깨졌으면 last_backup.pt로 이어간다 (최대 1 에폭 손실). 끝났으면 새로 시작한다.
- 이어서 할 학습이 있는데 --epochs 등 학습 옵션을 주면, 옵션이 무시되지 않도록 멈추고 안내한다.

사용:
    uv run python scripts/train.py                              # 새 학습, 또는 중단된 학습 이어서
    uv run python scripts/train.py --new                        # 중단된 학습이 있어도 새로 시작
    uv run python scripts/train.py --epochs 1 --fraction 0.1   # 시간 측정용 짧은 실행
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

BAR_WIDTH = 30
BACKUP_NAME = "last_backup.pt"


@dataclass
class EpochStatus:
    epoch: int  # 끝난 에폭 (1부터)
    epochs: int
    elapsed: float  # 학습 시작부터 초
    lr: float
    lr_max: float  # 지금까지 가장 컸던 학습률 (막대 기준)
    train_box_loss: float | None
    val_box_loss: float | None
    precision: float
    recall: float
    map50: float
    map50_95: float
    best_epoch: int
    best_map50_95: float
    patience: int
    start_epoch: int = 0  # 이어서 학습한 경우 이번 실행 전에 끝난 에폭 수 (elapsed는 이번 실행분)


def bar(frac: float, width: int = BAR_WIDTH) -> str:
    frac = min(max(frac, 0.0), 1.0)
    filled = round(frac * width)
    return "█" * filled + "░" * (width - filled)


def hms(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def render(s: EpochStatus) -> str:
    per_epoch = s.elapsed / max(s.epoch - s.start_epoch, 1)
    remaining = per_epoch * (s.epochs - s.epoch)
    no_improve = s.epoch - s.best_epoch
    # val loss는 YOLO26 검증에서 nan으로 나올 수 있다. 얼리 스톱과 best.pt는 mAP50-95 기준이라 영향 없음
    loss = lambda v: f"{v:.3f}" if v is not None and math.isfinite(v) else "-"  # noqa: E731
    lines = [
        f"━━━━━━━━━━ 에폭 {s.epoch}/{s.epochs} ━━━━━━━━━━",
        f"진행     [{bar(s.epoch / s.epochs)}] {s.epoch / s.epochs:6.1%}   경과 {hms(s.elapsed)}   "
        f"남은 예상 {hms(remaining)} (에폭당 {hms(per_epoch)})",
        f"학습률   [{bar(s.lr / s.lr_max if s.lr_max else 0)}] {s.lr:.6f} (최대 {s.lr_max:.6f})",
        f"손실     train box {loss(s.train_box_loss)}   val box {loss(s.val_box_loss)}",
        f"검증     P {s.precision:.3f}   R {s.recall:.3f}   mAP50 {s.map50:.3f}   mAP50-95 {s.map50_95:.3f}",
        f"최고     mAP50-95 {s.best_map50_95:.3f} (에폭 {s.best_epoch}) → best.pt",
        f"얼리스톱 [{bar(no_improve / s.patience if s.patience else 0, 10)}] {no_improve}/{s.patience} 에폭째 개선 없음",
    ]
    return "\n".join(lines)


def render_final(s: EpochStatus, best_path: str) -> str:
    return "\n".join(
        [
            "━━━━━━━━━━ 학습 끝: 최종 평가 ━━━━━━━━━━",
            f"모델     {best_path} (에폭 {s.best_epoch}의 가중치)",
            f"검증     P {s.precision:.3f}   R {s.recall:.3f}   mAP50 {s.map50:.3f}   mAP50-95 {s.map50_95:.3f}",
            f"총 소요  {hms(s.elapsed)}",
        ]
    )


class Dashboard:
    """Ultralytics 콜백. 에폭이 끝날 때마다 현황판을 출력한다."""

    def __init__(self):
        self.start = None
        self.lr_max = 0.0
        self.in_epoch = False
        self.start_epoch = 0

    def on_train_start(self, trainer) -> None:
        self.start = time.time()
        self.start_epoch = trainer.start_epoch

    def status(self, trainer) -> EpochStatus:
        m = trainer.metrics or {}
        losses = trainer.label_loss_items(trainer.tloss) if trainer.tloss is not None else {}
        lr = float(trainer.lr.get("lr/pg0", 0.0))
        self.lr_max = max(self.lr_max, lr)
        return EpochStatus(
            epoch=trainer.epoch + 1,
            epochs=trainer.epochs,
            elapsed=time.time() - self.start,
            lr=lr,
            lr_max=self.lr_max,
            train_box_loss=losses.get("train/box_loss"),
            val_box_loss=m.get("val/box_loss"),
            precision=m.get("metrics/precision(B)", 0.0),
            recall=m.get("metrics/recall(B)", 0.0),
            map50=m.get("metrics/mAP50(B)", 0.0),
            map50_95=m.get("metrics/mAP50-95(B)", 0.0),
            best_epoch=trainer.stopper.best_epoch,
            best_map50_95=trainer.stopper.best_fitness or 0.0,
            patience=trainer.args.patience,
            start_epoch=self.start_epoch,
        )

    def on_train_epoch_start(self, trainer) -> None:
        self.in_epoch = True

    def on_fit_epoch_end(self, trainer) -> None:
        """학습 루프가 끝난 뒤 best.pt 최종 평가에서도 한 번 더 불리므로, 에폭 시작 없이 불리면 최종 결과로 출력한다."""
        s = self.status(trainer)
        text = render(s) if self.in_epoch else render_final(s, str(trainer.best))
        self.in_epoch = False
        print("\n" + text + "\n", flush=True)

    def register(self, model) -> None:
        model.add_callback("on_train_start", self.on_train_start)
        model.add_callback("on_train_epoch_start", self.on_train_epoch_start)
        model.add_callback("on_fit_epoch_end", self.on_fit_epoch_end)


def atomic_copy(src: Path, dst: Path) -> None:
    """임시 파일에 복사한 뒤 이름을 바꾼다. 도중에 꺼져도 dst는 이전 내용 그대로다."""
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def backup_last(trainer) -> None:
    """on_model_save 콜백. 방금 저장된 last.pt를 last_backup.pt로 복사한다."""
    atomic_copy(Path(trainer.last), Path(trainer.last).with_name(BACKUP_NAME))


def remove_backup(trainer) -> None:
    """on_train_end 콜백. 끝난 학습의 백업이 재개 대상으로 잡히지 않게 지운다."""
    Path(trainer.last).with_name(BACKUP_NAME).unlink(missing_ok=True)


def _load_ckpt(path: Path) -> dict:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def _resumable(ckpt: dict) -> bool:
    """옵티마이저가 남아 있고 목표 에폭 전에 멈춘 체크포인트. 학습이 끝나면 Ultralytics가 epoch을 -1로 지운다."""
    epoch = ckpt.get("epoch", -1)
    return epoch >= 0 and ckpt.get("optimizer") is not None and epoch + 1 < ckpt.get("train_args", {}).get("epochs", 0)


def _try_load(path: Path, load) -> dict | None:
    """없거나 깨졌으면 (저장 중 전원이 꺼진 경우) None."""
    try:
        return load(path)
    except Exception:
        return None


def _last_saved(run: Path) -> float:
    files = [f for f in (run / "weights" / "last.pt", run / "weights" / BACKUP_NAME) if f.exists()]
    return max((f.stat().st_mtime for f in files), default=0.0)


def find_resumable(project: Path, name: str, load=_load_ckpt) -> Path | None:
    """project/name, name2, name3 ... 중 가장 최근에 저장된 학습이 끝나지 않았으면 그 체크포인트 경로.

    가장 최근 학습만 본다. --new로 버리고 새로 시작한 예전 학습이 나중에 되살아나지 않게 하기 위해서다.
    """
    runs = [d for d in project.glob(f"{name}*") if re.fullmatch(re.escape(name) + r"\d*", d.name)]
    if not runs:
        return None
    weights = max(runs, key=_last_saved) / "weights"
    last = _try_load(weights / "last.pt", load)
    if last is not None:
        return weights / "last.pt" if _resumable(last) else None  # 정상적으로 읽혔는데 재개 대상이 아니면 끝난 학습
    backup = _try_load(weights / BACKUP_NAME, load)
    return weights / BACKUP_NAME if backup is not None and _resumable(backup) else None


def check_resume_overrides(ckpt: Path | None, overrides: dict) -> None:
    """재개는 체크포인트의 설정을 그대로 쓴다. 학습 설정 옵션을 줬으면 조용히 무시하지 않고 멈춘다."""
    given = [f"--{k}" for k in ("epochs", "fraction", "batch", "imgsz") if overrides.get(k) is not None]
    if ckpt and given:
        raise SystemExit(
            f"중단된 학습이 있습니다: {ckpt}\n"
            f"이어서 하면 {', '.join(given)} 옵션은 적용되지 않습니다. "
            "이어서 하려면 옵션 없이 실행하고, 옵션대로 새로 시작하려면 --new를 붙이세요."
        )


def train_args(cfg: dict, overrides: dict) -> dict:
    """train.yaml에서 model을 빼고, 지정한 CLI 값으로 덮어쓴다.

    project는 절대 경로로 바꾼다. 상대 경로면 Ultralytics가 전역 설정 runs_dir 아래에 저장해서
    다른 프로젝트 폴더에 결과가 쌓일 수 있다.
    """
    args = {k: v for k, v in cfg.items() if k != "model"}
    args.update({k: v for k, v in overrides.items() if v is not None})
    if "project" in args:
        args["project"] = str(Path(args["project"]).resolve())
    return args


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--fraction", type=float, help="train 데이터 중 이 비율만 사용 (시간 측정용)")
    ap.add_argument("--batch", type=int)
    ap.add_argument("--imgsz", type=int)
    ap.add_argument("--name", help="runs/ 아래 결과 폴더 이름")
    ap.add_argument("--new", action="store_true", help="중단된 학습이 있어도 이어서 하지 않고 새로 시작")
    args = ap.parse_args(argv)

    from ultralytics import YOLO

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    overrides = {k: getattr(args, k) for k in ("epochs", "fraction", "batch", "imgsz", "name")}
    targs = train_args(cfg, overrides)
    ckpt = None if args.new else find_resumable(Path(targs["project"]), targs["name"])
    check_resume_overrides(ckpt, overrides)

    model = YOLO(str(ckpt) if ckpt else cfg["model"])
    Dashboard().register(model)
    model.add_callback("on_model_save", backup_last)
    model.add_callback("on_train_end", remove_backup)
    if ckpt:
        # 설정은 체크포인트에 저장된 것을 그대로 쓴다 (Ultralytics resume 규칙)
        print(f"중단된 학습을 이어서 합니다: {ckpt}", flush=True)
        model.train(resume=True)
    else:
        model.train(**targs)


if __name__ == "__main__":
    main()
