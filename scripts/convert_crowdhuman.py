"""CrowdHuman odgt → YOLO 데이터셋 변환 (README "데이터 처리").

- 유효한 person의 full box → 클래스 0 `person`. 이미지 경계로 자르고, 자른 뒤 크기가 없으면 버린다.
- `mask` 영역과 `extra.ignore == 1`인 사람은 라벨에 넣지 않고 이미지에서 회색으로 칠한다.
  라벨만 빼면 모델이 그 사람들(주로 멀리 있는 작은 군중)을 배경으로 배우기 때문이다.
  칠한 영역 안에 있는 유효 person의 visible box는 원래 픽셀로 되돌린다.
- 칠할 영역이 없는 이미지는 하드링크(안 되면 복사)로 둔다.
- 거리별 recall 평가(scripts/evaluate_count.py)에 필요한 visible box는 원본 odgt에서 다시 읽는다.

사용:
    uv run python scripts/convert_crowdhuman.py --src <CrowdHuman 폴더>

<CrowdHuman 폴더>에는 annotation_train.odgt, annotation_val.odgt와 이미지(*.jpg, 하위 폴더 어디든)가 있어야 한다.
결과: <dst>/images/{train,val}/ch_<ID>.jpg, <dst>/labels/{train,val}/ch_<ID>.txt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

FILL_VALUE = 114  # Ultralytics letterbox 여백과 같은 회색
JPEG_QUALITY = 95
PREFIX = "ch_"  # 다른 데이터셋을 섞을 때 파일명이 겹치지 않게


@dataclass
class ImageAnn:
    id: str
    persons: list[tuple[list[float], list[float]]]  # 유효 person의 (fbox, vbox), xywh
    ignores: list[list[float]]  # 칠할 영역, xywh


def parse_odgt(path: Path) -> list[ImageAnn]:
    anns = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            persons, ignores = [], []
            for b in d["gtboxes"]:
                ignored = b["tag"] != "person" or b.get("extra", {}).get("ignore", 0) == 1
                if ignored:
                    ignores.append(b.get("vbox") or b["fbox"])
                else:
                    persons.append((b["fbox"], b["vbox"]))
            anns.append(ImageAnn(d["ID"], persons, ignores))
    return anns


def _clip_xyxy(box: list[float], w: int, h: int) -> tuple[float, float, float, float]:
    x, y, bw, bh = box
    return max(0.0, x), max(0.0, y), min(float(w), x + bw), min(float(h), y + bh)


def _pixel_rect(box: list[float], w: int, h: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = _clip_xyxy(box, w, h)
    return math.floor(x0), math.floor(y0), math.ceil(x1), math.ceil(y1)


def yolo_labels(ann: ImageAnn, w: int, h: int) -> list[str]:
    lines = []
    for fbox, _ in ann.persons:
        x0, y0, x1, y1 = _clip_xyxy(fbox, w, h)
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
        lines.append(f"0 {cx:.6f} {cy:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
    return lines


def ignore_mask(ann: ImageAnn, w: int, h: int) -> np.ndarray | None:
    """칠할 픽셀. 칠할 게 없으면 None."""
    if not ann.ignores:
        return None
    mask = np.zeros((h, w), dtype=bool)
    for box in ann.ignores:
        x0, y0, x1, y1 = _pixel_rect(box, w, h)
        mask[y0:y1, x0:x1] = True
    for _, vbox in ann.persons:
        x0, y0, x1, y1 = _pixel_rect(vbox, w, h)
        mask[y0:y1, x0:x1] = False
    return mask if mask.any() else None


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.unlink(missing_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def convert_image(ann: ImageAnn, src_img: Path, img_dir: Path, lbl_dir: Path) -> tuple[int, bool]:
    """(라벨 수, 칠했는지)."""
    img = cv2.imread(str(src_img))
    if img is None:
        raise ValueError(f"이미지를 읽을 수 없음: {src_img}")
    h, w = img.shape[:2]
    name = PREFIX + ann.id
    labels = yolo_labels(ann, w, h)
    (lbl_dir / f"{name}.txt").write_text("".join(f"{s}\n" for s in labels), encoding="utf-8")

    dst_img = img_dir / f"{name}.jpg"
    mask = ignore_mask(ann, w, h)
    if mask is None:
        _link_or_copy(src_img, dst_img)
    else:
        img[mask] = FILL_VALUE
        # 이전 실행이 하드링크를 남겼으면 그대로 쓸 때 원본 CrowdHuman 이미지가 덮어써진다. 먼저 링크를 끊는다
        dst_img.unlink(missing_ok=True)
        if not cv2.imwrite(str(dst_img), img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
            raise OSError(f"이미지를 저장할 수 없음: {dst_img}")
    return len(labels), mask is not None


def index_images(src: Path) -> dict[str, Path]:
    """ID → 이미지 경로. 같은 ID가 여러 폴더에 있으면 처음 찾은 것."""
    index: dict[str, Path] = {}
    for p in sorted(src.rglob("*.jpg")):
        index.setdefault(p.stem, p)
    return index


def convert_split(split: str, src: Path, dst: Path, index: dict[str, Path], workers: int) -> dict[str, int]:
    anns = parse_odgt(src / f"annotation_{split}.odgt")
    img_dir, lbl_dir = dst / "images" / split, dst / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    found = [a for a in anns if a.id in index]
    stats = {
        "images": len(found),
        "missing": len(anns) - len(found),
        "persons": 0,
        "ignored_boxes": sum(len(a.ignores) for a in found),
        "masked_images": 0,
    }
    with ThreadPoolExecutor(workers) as pool:
        for n, masked in pool.map(lambda a: convert_image(a, index[a.id], img_dir, lbl_dir), found):
            stats["persons"] += n
            stats["masked_images"] += masked
    return stats


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, default=Path("datasets/crowdhuman"), help="CrowdHuman 폴더")
    ap.add_argument("--dst", type=Path, default=Path("datasets/person"), help="configs/data.yaml의 path")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)

    index = index_images(args.src)
    print(f"이미지 {len(index)}장 찾음: {args.src}")
    for split in args.splits:
        s = convert_split(split, args.src, args.dst, index, args.workers)
        print(
            f"[{split}] 이미지 {s['images']}장 (없음 {s['missing']}), person {s['persons']}개, "
            f"제외 영역 {s['ignored_boxes']}개, 회색 처리한 이미지 {s['masked_images']}장"
        )
        if s["missing"]:
            print(f"[{split}] 경고: 주석은 있는데 이미지가 없는 항목 {s['missing']}개를 건너뜀")


if __name__ == "__main__":
    main()
