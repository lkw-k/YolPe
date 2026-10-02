"""person 박스 높이(px) 구간별 recall 측정 (README "평가 방식").

원거리에서 사람을 인식할 수 있는지 없는지를 가른다. CrowdHuman val의 원본 이미지와 odgt를 쓴다.
- 검출 성공: 검출 박스가 GT visible box와 IoU >= iou (COCO 모델은 보이는 부분만 잡기 때문)
- 높이 구간: GT full box 높이. 가려져도 거리를 반영하므로. 추론 입력 해상도 기준 px로 환산한다.
- 최소 검출 크기: 그 구간부터 위로 모든 구간의 recall이 recall_target 이상인 가장 작은 높이.
  가림 효과를 빼기 위해 가시율(visible/full 면적비) >= min_visibility인 사람으로만 정한다. 전체 표도 같이 낸다.
- ignore/mask로 표시된 사람은 GT에서 뺀다

사용:
    uv run python scripts/evaluate_count.py --src <CrowdHuman 폴더> [--model yolo26s.pt] [--limit 200]
결과: 표 출력 + runs/eval/recall_<모델>_<imgsz>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_crowdhuman import ImageAnn, index_images, parse_odgt  # noqa: E402


@dataclass
class HeightBin:
    lo: float
    hi: float  # inf면 열린 구간
    gt: int
    detected: int

    @property
    def recall(self) -> float | None:
        return self.detected / self.gt if self.gt else None


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a (N,4), b (M,4) xyxy → (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def match_gt(preds: np.ndarray, scores: np.ndarray, gts: np.ndarray, iou_thr: float) -> np.ndarray:
    """신뢰도 높은 검출부터 아직 안 잡힌 GT 중 IoU가 가장 큰 것에 1:1로 배정. GT별 검출 여부를 반환."""
    matched = np.zeros(len(gts), dtype=bool)
    ious = iou_matrix(preds, gts)
    for i in np.argsort(-scores, kind="stable"):
        if len(gts) == 0:
            break
        cand = np.where(matched, -1.0, ious[i])
        j = int(np.argmax(cand))
        if cand[j] >= iou_thr:
            matched[j] = True
    return matched


def gt_arrays(ann: ImageAnn, w: int, h: int, imgsz: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(이미지 경계로 자른 visible box xyxy, 입력 해상도 기준 full box 높이, 가시율)."""
    scale = imgsz / max(w, h)  # Ultralytics letterbox: 긴 변을 imgsz에 맞춤
    vboxes, heights, vis = [], [], []
    for fbox, (x, y, bw, bh) in ann.persons:
        vboxes.append([max(0, x), max(0, y), min(w, x + bw), min(h, y + bh)])
        heights.append(fbox[3] * scale)
        vis.append(bw * bh / max(fbox[2] * fbox[3], 1e-9))
    return np.array(vboxes, dtype=float).reshape(-1, 4), np.array(heights, dtype=float), np.array(vis, dtype=float)


def recall_by_height(heights: np.ndarray, matched: np.ndarray, edges: list[float]) -> list[HeightBin]:
    bounds = list(edges) + [float("inf")]
    bins = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        sel = (heights >= lo) & (heights < hi)
        bins.append(HeightBin(lo, hi, int(sel.sum()), int(matched[sel].sum())))
    return bins


def min_detectable_height(bins: list[HeightBin], target: float, min_samples: int) -> float | None:
    """위 구간부터 내려오며 recall >= target이 이어지는 마지막 구간의 하한. 가장 큰 구간부터 미달이면 None."""
    result = None
    for b in reversed(bins):
        if b.gt < min_samples:
            continue
        if b.recall < target:
            break
        result = b.lo
    return result


def _predict(model, paths: list[Path], imgsz: int, det_cfg, device, batch: int):
    """Ultralytics는 경로 리스트 전체를 한 배치로 추론하므로(4천 장이면 GPU 메모리 부족) batch 장씩 나눠 넘긴다."""
    for i in range(0, len(paths), batch):
        yield from model.predict(
            source=[str(p) for p in paths[i : i + batch]],
            imgsz=imgsz,
            conf=det_cfg.person_conf,
            max_det=det_cfg.max_det,
            classes=[0],
            device=device,
            verbose=False,
        )


def evaluate(
    model, anns: list[ImageAnn], paths: list[Path], imgsz: int, det_cfg, iou_thr: float, device, batch: int = 8
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """모든 GT의 (입력 해상도 기준 높이, 검출 여부, 가시율)."""
    all_h, all_m, all_v = [], [], []
    for ann, r in zip(anns, _predict(model, paths, imgsz, det_cfg, device, batch), strict=True):
        h, w = r.orig_shape
        gts, heights, vis = gt_arrays(ann, w, h, imgsz)
        boxes = r.boxes
        matched = match_gt(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), gts, iou_thr)
        all_h.append(heights)
        all_m.append(matched)
        all_v.append(vis)
    return np.concatenate(all_h), np.concatenate(all_m), np.concatenate(all_v)


def _fmt_bin(b: HeightBin) -> str:
    rng = f"{b.lo:>4.0f}~{b.hi:<4.0f}" if b.hi != float("inf") else f"{b.lo:>4.0f}~    "
    rec = f"{b.recall:6.3f}" if b.recall is not None else "     -"
    return f"  {rng} px | GT {b.gt:6d} | 검출 {b.detected:6d} | recall {rec}"


def _print_table(title: str, bins: list[HeightBin], n: int, recall: float) -> None:
    print(f"  {title}: GT {n}명, 전체 recall {recall:.3f}")
    for b in bins:
        print(_fmt_bin(b))


def _bins_json(bins: list[HeightBin]) -> list[dict]:
    return [{**asdict(b), "hi": None if b.hi == float("inf") else b.hi, "recall": b.recall} for b in bins]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, default=Path("datasets/crowdhuman"), help="CrowdHuman 폴더")
    ap.add_argument("--split", default="val")
    ap.add_argument("--model", default="yolo26s.pt", help="COCO 사전학습 가중치 또는 파인튜닝 모델")
    ap.add_argument("--config", type=Path, default=Path("configs/eval.yaml"))
    ap.add_argument("--imgsz", type=int, nargs="+", help="지정하면 eval.yaml의 값 대신 사용")
    ap.add_argument("--batch", type=int, default=8, help="추론 배치 크기 (GPU 메모리에 맞게)")
    ap.add_argument("--limit", type=int, help="앞에서부터 N장만 (빠른 확인용)")
    ap.add_argument("--out", type=Path, default=Path("runs/eval"))
    args = ap.parse_args(argv)

    from ultralytics import YOLO

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    det_cfg = SimpleNamespace(**cfg["detector"])
    model = YOLO(args.model)
    if model.names.get(0) != "person":
        raise SystemExit(f"모델의 0번 클래스가 person이 아님: {model.names.get(0)}")

    index = index_images(args.src)
    anns = [a for a in parse_odgt(args.src / f"annotation_{args.split}.odgt") if a.id in index]
    if args.limit:
        anns = anns[: args.limit]
    paths = [index[a.id] for a in anns]

    args.out.mkdir(parents=True, exist_ok=True)
    for imgsz in args.imgsz or cfg["imgsz"]:
        heights, matched, vis = evaluate(model, anns, paths, imgsz, det_cfg, cfg["iou"], det_cfg.device, args.batch)
        clear = vis >= cfg["min_visibility"]
        bins_all = recall_by_height(heights, matched, cfg["height_bins"])
        bins = recall_by_height(heights[clear], matched[clear], cfg["height_bins"])
        min_h = min_detectable_height(bins, cfg["recall_target"], cfg["min_samples"])

        print(f"\n[{Path(args.model).stem}, imgsz {imgsz}] 이미지 {len(anns)}장")
        _print_table("전체", bins_all, len(heights), float(matched.mean()))
        _print_table(f"가시율 >= {cfg['min_visibility']}", bins, int(clear.sum()), float(matched[clear].mean()))
        print(
            f"  최소 검출 크기 (가시율 >= {cfg['min_visibility']}, recall >= {cfg['recall_target']}): "
            + (f"{min_h:.0f}px" if min_h is not None else "없음")
        )

        out = args.out / f"recall_{Path(args.model).stem}_{imgsz}.json"
        out.write_text(
            json.dumps(
                {
                    "model": args.model,
                    "imgsz": imgsz,
                    "split": args.split,
                    "images": len(anns),
                    "person_conf": det_cfg.person_conf,
                    "max_det": det_cfg.max_det,
                    **{k: cfg[k] for k in ("iou", "recall_target", "min_samples", "min_visibility")},
                    "min_detectable_height": min_h,
                    "all": {"gt": len(heights), "recall": float(matched.mean()), "bins": _bins_json(bins_all)},
                    "visible": {"gt": int(clear.sum()), "recall": float(matched[clear].mean()), "bins": _bins_json(bins)},
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"  저장: {out}")


if __name__ == "__main__":
    main()
