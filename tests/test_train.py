"""학습 스크립트의 설정 병합과 현황판 (README "학습"). 실제 학습 없이 가짜 trainer로 확인한다."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train as tr  # noqa: E402


def test_bar_fills_proportionally_and_clamps():
    assert tr.bar(0.5, 10) == "█████░░░░░"
    assert tr.bar(0, 4) == "░░░░"
    assert tr.bar(1.7, 4) == "████"


def test_hms():
    assert tr.hms(3725) == "1:02:05"


def _status(**kw):
    base = dict(
        epoch=10, epochs=50, elapsed=1000.0, lr=0.005, lr_max=0.01, train_box_loss=1.2, val_box_loss=1.5,
        precision=0.8, recall=0.6, map50=0.7, map50_95=0.4, best_epoch=7, best_map50_95=0.42, patience=10,
    )
    return tr.EpochStatus(**{**base, **kw})


def test_render_progress_eta_lr_and_early_stop():
    text = tr.render(_status())
    assert f"[{tr.bar(0.2)}]  20.0%" in text  # 10/50 에폭
    assert "남은 예상 1:06:40" in text  # 에폭당 100초 × 남은 40 에폭
    assert f"[{tr.bar(0.5)}] 0.005000" in text  # 학습률은 최대값 대비
    assert "3/10 에폭째 개선 없음" in text  # 최고가 에폭 7
    assert "mAP50-95 0.420 (에폭 7)" in text


def test_render_handles_missing_or_nan_loss():
    assert "val box -" in tr.render(_status(val_box_loss=None))
    assert "val box -" in tr.render(_status(val_box_loss=float("nan")))  # YOLO26 검증 손실이 nan으로 올 때


def _trainer(epoch, lr, metrics=None):
    return SimpleNamespace(
        epoch=epoch,
        epochs=50,
        metrics=metrics or {"metrics/mAP50-95(B)": 0.3, "val/box_loss": 1.1},
        tloss=[1.0],
        label_loss_items=lambda t: {"train/box_loss": t[0]},
        lr={"lr/pg0": lr},
        stopper=SimpleNamespace(best_epoch=epoch + 1, best_fitness=0.3),
        args=SimpleNamespace(patience=10),
    )


def test_dashboard_reads_trainer_and_tracks_max_lr():
    d = tr.Dashboard()
    d.on_train_start(SimpleNamespace(start_epoch=0))
    s1 = d.status(_trainer(0, 0.002))  # 워밍업 중
    s2 = d.status(_trainer(1, 0.01))
    s3 = d.status(_trainer(2, 0.004))  # 감소 구간
    assert (s1.epoch, s1.lr_max) == (1, 0.002)
    assert s3.lr_max == 0.01 and s3.lr == 0.004
    assert s2.map50_95 == 0.3 and s2.val_box_loss == 1.1 and s2.train_box_loss == 1.0
    assert s2.precision == 0.0  # 없는 지표는 0


def test_final_eval_call_prints_summary_not_epoch(capsys):
    """학습 루프가 끝난 뒤 best.pt 평가에서 on_fit_epoch_end가 한 번 더 불린다 (에폭 시작 없이)."""
    d = tr.Dashboard()
    d.on_train_start(SimpleNamespace(start_epoch=0))
    t = _trainer(0, 0.01)
    t.epochs, t.best = 1, "runs/x/weights/best.pt"

    d.on_train_epoch_start(t)
    d.on_fit_epoch_end(t)
    t.epoch = 1  # Ultralytics가 최종 평가 때 epoch를 1 올린다
    d.on_fit_epoch_end(t)

    out = capsys.readouterr().out
    assert "에폭 1/1" in out
    assert "에폭 2/1" not in out
    assert "학습 끝: 최종 평가" in out and "runs/x/weights/best.pt" in out


def test_eta_after_resume_uses_only_this_session_epochs():
    # 20 에폭까지 끝난 체크포인트에서 이어서 2 에폭(200초)을 돌림 → 에폭당 100초, 남은 28 에폭
    text = tr.render(_status(epoch=22, elapsed=200.0, start_epoch=20))
    assert "에폭당 0:01:40" in text and "남은 예상 0:46:40" in text


def test_atomic_copy_and_backup_callbacks(tmp_path):
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "last.pt").write_bytes(b"epoch3")
    trainer = SimpleNamespace(last=weights / "last.pt")

    tr.backup_last(trainer)
    assert (weights / tr.BACKUP_NAME).read_bytes() == b"epoch3"
    assert not list(weights.glob("*.tmp"))  # 임시 파일이 남지 않음

    tr.remove_backup(trainer)
    assert not (weights / tr.BACKUP_NAME).exists()


RUNNING = {"epoch": 4, "optimizer": {}, "train_args": {"epochs": 50}}  # 에폭 5까지 끝남
FINISHED = {"epoch": -1, "optimizer": None, "train_args": {"epochs": 50}}  # strip_optimizer 이후


def _make_runs(tmp_path, runs: dict):
    """runs: {폴더명: {파일명: 체크포인트 dict 또는 "broken"}}. 뒤에 적은 것일수록 최근에 저장."""
    store = {}
    for i, (run, files) in enumerate(runs.items()):
        w = tmp_path / run / "weights"
        w.mkdir(parents=True)
        for fname, ckpt in files.items():
            p = w / fname
            p.write_bytes(b"x")
            os.utime(p, (1000 + i, 1000 + i))
            store[p] = ckpt

    def load(path):
        ckpt = store[Path(path)]  # 없으면 KeyError (파일 없음과 같은 취급)
        if ckpt == "broken":
            raise RuntimeError("PytorchStreamReader failed reading zip archive")
        return ckpt

    return load


def test_resume_picks_unfinished_run(tmp_path):
    load = _make_runs(tmp_path, {"YolPe": {"last.pt": FINISHED}, "YolPe2": {"last.pt": RUNNING}})
    assert tr.find_resumable(tmp_path, "YolPe", load) == tmp_path / "YolPe2" / "weights" / "last.pt"


def test_resume_skips_finished_run_even_if_stale_backup_exists(tmp_path):
    load = _make_runs(tmp_path, {"YolPe": {tr.BACKUP_NAME: RUNNING, "last.pt": FINISHED}})
    assert tr.find_resumable(tmp_path, "YolPe", load) is None


def test_resume_falls_back_to_backup_when_last_is_broken(tmp_path):
    load = _make_runs(tmp_path, {"YolPe": {tr.BACKUP_NAME: RUNNING, "last.pt": "broken"}})
    assert tr.find_resumable(tmp_path, "YolPe", load) == tmp_path / "YolPe" / "weights" / tr.BACKUP_NAME


def test_resume_ignores_other_run_names_and_final_epoch(tmp_path):
    done = {"epoch": 49, "optimizer": {}, "train_args": {"epochs": 50}}  # 마지막 에폭까지 저장됨 → 이어갈 게 없음
    load = _make_runs(tmp_path, {"timing": {"last.pt": RUNNING}, "YolPe": {"last.pt": done}})
    assert tr.find_resumable(tmp_path, "YolPe", load) is None
    assert tr.find_resumable(tmp_path / "nothing", "YolPe", load) is None


def test_abandoned_older_run_is_not_resumed_after_newer_run_finishes(tmp_path):
    # 중단된 YolPe를 --new로 버리고 YolPe2를 끝까지 돌린 뒤 → 새 학습을 시작해야 한다
    load = _make_runs(tmp_path, {"YolPe": {"last.pt": RUNNING}, "YolPe2": {"last.pt": FINISHED}})
    assert tr.find_resumable(tmp_path, "YolPe", load) is None


def test_resume_refuses_training_options_it_would_ignore():
    ckpt = Path("runs/YolPe/weights/last.pt")
    with pytest.raises(SystemExit, match="--epochs, --fraction"):
        tr.check_resume_overrides(ckpt, {"epochs": 3, "fraction": 0.1, "batch": None, "imgsz": None, "name": None})
    tr.check_resume_overrides(ckpt, {"epochs": None, "fraction": None, "batch": None, "imgsz": None, "name": "x"})
    tr.check_resume_overrides(None, {"epochs": 3})  # 재개할 게 없으면 옵션대로 새 학습


def test_train_args_drop_model_and_apply_overrides():
    cfg = yaml.safe_load((ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))
    args = tr.train_args(cfg, {"epochs": 1, "fraction": 0.1, "batch": None})
    assert "model" not in args
    assert args["epochs"] == 1 and args["fraction"] == 0.1
    assert args["batch"] == cfg["batch"]  # None은 덮어쓰지 않음
    assert Path(args["project"]).is_absolute()  # 전역 runs_dir 아래로 새지 않게


def test_train_yaml_guards_against_overfitting():
    cfg = yaml.safe_load((ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))
    assert cfg["epochs"] == 50
    assert 0 < cfg["patience"] < cfg["epochs"]  # 얼리 스톱이 실제로 동작할 수 있는 값
    assert cfg["max_det"] == 1000
