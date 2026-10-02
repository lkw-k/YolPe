"""거리별 recall 평가 (README "평가 방식"). 모델 없이 가짜 예측으로 확인한다."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_count as ev  # noqa: E402
from convert_crowdhuman import ImageAnn  # noqa: E402

INF = float("inf")


def test_iou_matrix():
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], float)
    np.testing.assert_allclose(ev.iou_matrix(a, b), [[1.0, 50 / 150, 0.0]])


def test_one_detection_matches_only_one_gt():
    gts = np.array([[0, 0, 10, 10], [1, 0, 11, 10]], float)  # 거의 겹친 두 사람
    preds = np.array([[0, 0, 10, 10]], float)
    assert ev.match_gt(preds, np.array([0.9]), gts, 0.5).tolist() == [True, False]


def test_higher_confidence_detection_takes_best_gt_first():
    gts = np.array([[0, 0, 10, 10]], float)
    preds = np.array([[2, 0, 12, 10], [0, 0, 10, 10]], float)  # 두 번째가 더 정확하지만 신뢰도가 낮음
    matched = ev.match_gt(preds, np.array([0.9, 0.5]), gts, 0.5)
    assert matched.tolist() == [True]


def test_second_detection_takes_remaining_gt():
    gts = np.array([[0, 0, 10, 10], [2, 0, 12, 10]], float)
    preds = np.array([[0, 0, 10, 10], [1, 0, 11, 10]], float)
    # 첫 검출이 GT0을 가져가면 두 번째 검출은 GT1로 가야 한다
    assert ev.match_gt(preds, np.array([0.9, 0.8]), gts, 0.5).tolist() == [True, True]


def test_below_iou_threshold_not_matched_and_empty_inputs():
    gts = np.array([[0, 0, 10, 10]], float)
    assert ev.match_gt(np.array([[5, 5, 15, 15]], float), np.array([0.9]), gts, 0.5).tolist() == [False]
    assert ev.match_gt(np.zeros((0, 4)), np.zeros(0), gts, 0.5).tolist() == [False]
    assert ev.match_gt(np.array([[0, 0, 1, 1]], float), np.array([0.9]), np.zeros((0, 4)), 0.5).tolist() == []


def test_gt_arrays_scale_by_long_side_and_clip_visible_box():
    ann = ImageAnn("x", persons=[([10, 10, 20, 400], [-5, 10, 20, 50])], ignores=[])
    vboxes, heights, vis = ev.gt_arrays(ann, w=2000, h=1000, imgsz=1000)
    assert vboxes.tolist() == [[0, 10, 15, 60]]
    assert heights.tolist() == [200.0]  # 400px × (1000 / 2000), full box 기준
    assert vis.tolist() == [0.125]  # visible 20×50 / full 20×400


def test_recall_by_height_with_open_last_bin():
    heights = np.array([5, 9, 15, 300, 500])
    matched = np.array([False, True, True, True, False])
    bins = ev.recall_by_height(heights, matched, [0, 8, 16])
    assert [(b.lo, b.hi, b.gt, b.detected) for b in bins] == [(0, 8, 1, 0), (8, 16, 2, 2), (16, INF, 2, 1)]
    assert bins[1].recall == 1.0


def _bins(*rows):
    return [ev.HeightBin(lo, hi, gt, det) for lo, hi, gt, det in rows]


def test_min_height_is_lowest_bin_of_unbroken_run_from_top():
    bins = _bins((0, 8, 100, 90), (8, 16, 100, 50), (16, 32, 100, 80), (32, INF, 100, 95))
    # 0~8이 0.9라도 8~16에서 끊기므로 16
    assert ev.min_detectable_height(bins, 0.7, 50) == 16


def test_min_height_skips_small_sample_bins():
    bins = _bins((0, 8, 100, 10), (8, 16, 3, 0), (16, INF, 100, 95))
    assert ev.min_detectable_height(bins, 0.7, 50) == 16
    bins = _bins((0, 8, 100, 80), (8, 16, 3, 0), (16, INF, 100, 95))
    assert ev.min_detectable_height(bins, 0.7, 50) == 0


def test_min_height_none_when_largest_bin_fails():
    assert ev.min_detectable_height(_bins((0, INF, 100, 10)), 0.7, 50) is None


class _Arr:
    def __init__(self, a):
        self.a = np.asarray(a, float)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class FakeModel:
    """이미지마다 정해 둔 박스를 돌려주는 가짜 YOLO."""

    def __init__(self, outputs, shape):
        self.outputs, self.shape, self.calls = iter(outputs), shape, []

    def predict(self, source, **kwargs):
        self.calls.append((source, kwargs))
        return [
            SimpleNamespace(
                orig_shape=self.shape,
                boxes=SimpleNamespace(xyxy=_Arr(np.reshape(boxes, (-1, 4))), conf=_Arr([0.9] * len(boxes))),
            )
            for boxes in (next(self.outputs) for _ in source)
        ]


def test_evaluate_collects_heights_and_matches_per_image():
    anns = [
        ImageAnn("a", [([0, 0, 10, 100], [0, 0, 10, 100]), ([50, 0, 10, 40], [50, 0, 10, 40])], []),
        ImageAnn("b", [([0, 0, 10, 20], [0, 0, 10, 20])], []),
    ]
    model = FakeModel([[[0, 0, 10, 100]], []], shape=(100, 200))  # 이미지 a의 첫 사람만 검출
    det = SimpleNamespace(person_conf=0.35, max_det=1000)
    heights, matched, vis = ev.evaluate(model, anns, [Path("a.jpg"), Path("b.jpg")], 400, det, 0.5, None, batch=1)

    assert heights.tolist() == [200, 80, 40]  # × 400/200
    assert matched.tolist() == [True, False, False]
    assert vis.tolist() == [1.0, 1.0, 1.0]
    # 경로 리스트 전체가 한 배치가 되지 않도록 batch 장씩 나눠 호출
    assert [src for src, _ in model.calls] == [["a.jpg"], ["b.jpg"]]
    kw = model.calls[0][1]
    assert kw["classes"] == [0] and kw["max_det"] == 1000 and kw["conf"] == 0.35
