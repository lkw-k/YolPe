"""CrowdHuman 변환 (README "데이터 처리"). 실제 데이터 없이 작은 가짜 odgt와 이미지로 확인한다."""

import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("convert_crowdhuman", ROOT / "scripts" / "convert_crowdhuman.py")
cc = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = cc  # dataclass가 모듈을 찾을 수 있게
_spec.loader.exec_module(cc)

W, H = 200, 100
BG = 200  # 원본 배경 밝기. 회색 처리 값(114)과 구분되게


def _box(tag, fbox, vbox, ignore=0):
    return {"tag": tag, "fbox": fbox, "vbox": vbox, "hbox": [0, 0, 1, 1], "extra": {"ignore": ignore}}


def _make_src(tmp_path, records):
    src = tmp_path / "ch"
    (src / "Images").mkdir(parents=True)
    with open(src / "annotation_val.odgt", "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
            cv2.imwrite(str(src / "Images" / f"{rec['ID']}.jpg"), np.full((H, W, 3), BG, np.uint8))
    return src


def _run(tmp_path, records):
    src = _make_src(tmp_path, records)
    dst = tmp_path / "out"
    cc.main(["--src", str(src), "--dst", str(dst), "--splits", "val", "--workers", "2"])
    return dst


def _labels(dst, id_):
    text = (dst / "labels" / "val" / f"ch_{id_}.txt").read_text(encoding="utf-8")
    return [list(map(float, line.split())) for line in text.splitlines()]


def test_full_box_becomes_normalized_person_label(tmp_path):
    dst = _run(tmp_path, [{"ID": "a,1", "gtboxes": [_box("person", [20, 10, 40, 80], [20, 10, 40, 50])]}])
    (cls, cx, cy, w, h), = _labels(dst, "a,1")
    assert cls == 0
    assert (cx, cy, w, h) == (0.2, 0.5, 0.2, 0.8)  # visible box가 아니라 full box


def test_box_outside_image_is_clipped_or_dropped(tmp_path):
    boxes = [
        _box("person", [-20, 50, 60, 100], [0, 50, 40, 50]),  # 왼쪽·아래로 넘침 → 자름
        _box("person", [250, 10, 30, 30], [250, 10, 30, 30]),  # 완전히 밖 → 버림
    ]
    dst = _run(tmp_path, [{"ID": "b", "gtboxes": boxes}])
    (_, cx, cy, w, h), = _labels(dst, "b")
    assert (cx, cy, w, h) == (0.1, 0.75, 0.2, 0.5)  # x 0~40, y 50~100


def test_ignored_regions_are_not_labeled_and_filled_gray(tmp_path):
    boxes = [
        _box("mask", [100, 0, 100, 100], [100, 0, 100, 100], ignore=1),
        _box("person", [0, 0, 40, 100], [0, 0, 40, 100], ignore=1),
        _box("person", [140, 20, 20, 60], [140, 20, 20, 40]),  # 제외 영역 안의 유효 person
    ]
    dst = _run(tmp_path, [{"ID": "c", "gtboxes": boxes}])

    assert len(_labels(dst, "c")) == 1
    img = cv2.imread(str(dst / "images" / "val" / "ch_c.jpg")).astype(int)
    assert abs(img[50, 20].mean() - cc.FILL_VALUE) < 5  # ignore person
    assert abs(img[90, 180].mean() - cc.FILL_VALUE) < 5  # mask
    assert abs(img[40, 150].mean() - BG) < 5  # 유효 person의 visible box는 되돌림
    assert abs(img[50, 70].mean() - BG) < 5  # 아무 박스도 없는 곳은 그대로


def test_rerun_with_new_ignore_region_does_not_touch_source_image(tmp_path):
    """1차 실행에서 하드링크로 둔 이미지를 2차 실행에서 회색 처리해도 원본은 그대로여야 한다."""
    src = _make_src(tmp_path, [{"ID": "f", "gtboxes": []}])  # 칠할 영역 없음 → 하드링크
    dst = tmp_path / "out"
    cc.main(["--src", str(src), "--dst", str(dst), "--splits", "val"])

    rec = {"ID": "f", "gtboxes": [_box("mask", [0, 0, 100, 100], [0, 0, 100, 100], ignore=1)]}
    (src / "annotation_val.odgt").write_text(json.dumps(rec) + "\n", encoding="utf-8")
    cc.main(["--src", str(src), "--dst", str(dst), "--splits", "val"])

    original = cv2.imread(str(src / "Images" / "f.jpg")).astype(int)
    converted = cv2.imread(str(dst / "images" / "val" / "ch_f.jpg")).astype(int)
    assert abs(original[50, 50].mean() - BG) < 5  # 원본은 그대로
    assert abs(converted[50, 50].mean() - cc.FILL_VALUE) < 5  # 결과만 회색


def test_image_without_people_gets_empty_label(tmp_path):
    dst = _run(tmp_path, [{"ID": "d", "gtboxes": []}])
    assert _labels(dst, "d") == []
    assert (dst / "images" / "val" / "ch_d.jpg").exists()


def test_missing_image_is_skipped(tmp_path, capsys):
    src = _make_src(tmp_path, [{"ID": "e", "gtboxes": []}])
    with open(src / "annotation_val.odgt", "a", encoding="utf-8") as f:
        f.write(json.dumps({"ID": "no_image", "gtboxes": []}) + "\n")
    dst = tmp_path / "out"
    cc.main(["--src", str(src), "--dst", str(dst), "--splits", "val"])

    assert not (dst / "labels" / "val" / "ch_no_image.txt").exists()
    assert "없음 1" in capsys.readouterr().out
