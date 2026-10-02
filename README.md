# YolPe: CCTV 원거리 사람 인식을 위한 YOLO26s 파인튜닝

COCO로 사전학습된 YOLO26s를 군중 데이터셋 [CrowdHuman](https://www.crowdhuman.org/)으로 파인튜닝해,
**멀리 있는 작은 사람과 서로 겹친 사람**을 더 잘 찾도록 만든 사람(`person`) 검출 모델의 학습 코드입니다.

- 모델 가중치와 상세 성능표: **[Hugging Face illimax/YolPe](https://huggingface.co/illimax/YolPe)**
- 이 저장소: 데이터 변환 → 학습 → 거리별 평가 코드

## 왜 만들었나

CCTV는 높은 곳에서 넓은 범위를 찍습니다. 그래서 같은 화면 안에서도 가까운 사람은 수백 px로 크게,
멀리 있는 사람은 수십 px로 작게 보이고, 사람이 많으면 서로 겹칩니다.
화면 속 사람 수를 세려면 이렇게 작고 겹친 사람까지 빠짐없이 찾아야 합니다.

COCO 사전학습 YOLO26s를 그대로 쓰면 가까운 사람은 잘 찾지만 작은 사람은 많이 놓칩니다.
아래 기준으로 재 보면 안정적으로 찾는 가장 작은 사람 키가 128px였습니다.
그래서 근거리 성능은 유지하면서 원거리 인식을 끌어올리는 것을 목표로 파인튜닝했습니다.

## 결과 요약

CrowdHuman val 4,370장, imgsz 1280 기준입니다. 지표마다 재는 조건이 달라서 수치를 직접 비교하면 안 됩니다.

| 지표 | conf | 정답 박스 |
|---|---|---|
| **거리별 인식률** (기준) | 0.35 | visible box, IoU 0.5 ([평가 방식](#평가-방식)) |
| 검출 지표 (P, R, mAP) | 0.371 (최대 F1 지점) | full box (Ultralytics 표준 검증) |

**거리별 인식률** (conf 0.35, 몸의 75% 이상이 보이는 사람 48,204명)

| 거리 (사람 박스 높이) | 기본 YOLO26s | YolPe |
|---|---|---|
| 원거리 (48px 미만) | 0.225 | **0.798** |
| 중거리 (48~128px) | 0.559 | **0.930** |
| 근거리 (128px 이상) | 0.852 | **0.969** |
| 전체 | 0.689 | **0.938** |
| 최소 인식 크기 (recall ≥ 0.7) | 128px | **24px** |

**최대 F1 지점** (conf 0.371, F1 0.91, best epoch 37)

- 검출 지표: P 0.940, R 0.884, mAP50 0.946, mAP50-95 0.660
- 같은 conf로 거리별 인식률을 다시 재도 COCO 대비 결과는 거의 같습니다: 원거리 0.212 → **0.789**, 전체 0.678 → **0.935**, 최소 인식 크기 128px → **24px**

가려진 사람을 포함한 결과, imgsz 960 결과, 구간별 상세 표는 [모델 카드](https://huggingface.co/illimax/YolPe)에 있습니다.

## 데이터 처리

`scripts/convert_crowdhuman.py`가 CrowdHuman의 odgt 주석을 Ultralytics YOLO 형식으로 바꿉니다.

- **라벨**: 유효한 사람의 full box(가려진 부분까지 포함한 전체 몸)를 클래스 `0: person`으로 씁니다.
  사람을 세는 것이 목적이라, 일부만 보여도 사람 한 명으로 잡게 합니다. full box가 이미지 밖으로 나가면 경계에서 자릅니다.
- **제외 영역은 회색으로 칠함**: CrowdHuman의 `mask` 영역과 `extra.ignore = 1`인 사람은 라벨에 넣지 않고,
  이미지에서 회색(114, Ultralytics letterbox 여백과 같은 값)으로 칠합니다.
  - `mask`에는 너무 작거나 빽빽해서 라벨을 못 단 군중이 들어 있습니다. 라벨만 빼면 모델이 그 사람들을
    "사람 없음"으로 배워서, 바로 우리가 원하는 원거리 검출이 약해집니다.
  - 칠한 영역 안에 있는 유효한 사람의 보이는 부분(visible box)은 원래 픽셀로 되돌립니다.
  - 대신 `mask`에 섞여 있는 포스터 속 인물이나 바닥 반사를 "사람 아님"으로 배울 기회도 잃습니다 (아래 한계 참고).
- 칠할 영역이 없는 이미지는 하드링크(안 되면 복사)로 둬서 디스크를 아낍니다. 다시 실행해도 원본 이미지는 바뀌지 않습니다.

train 15,000장으로 학습하고 val 4,370장으로 검증했습니다.

## 학습

`scripts/train.py`가 `configs/train.yaml`을 읽어 Ultralytics `train()`에 넘깁니다.

| 항목 | 값 | 이유 |
|---|---|---|
| model | yolo26s.pt (COCO) | |
| imgsz | 1280 | 작은 사람이 줄어들어 사라지지 않게 |
| epochs / patience | 50 / 10 | val mAP50-95가 10 에폭 동안 안 오르면 중단 |
| batch | 8 | RTX 4070 12GB 기준 |
| mosaic / close_mosaic | 1.0 / 10 | 마지막 10 에폭은 모자이크 없이 실제 이미지 분포로 마무리 |
| scale / fliplr | 0.5 / 0.5 | |
| flipud | 0.0 | CCTV는 시점이 고정이라 사람이 거꾸로 보일 일이 없음 |
| hsv_v | 0.4 | 실내 조명 변화 |
| max_det | 1000 | 혼잡 장면에서 인원이 잘리지 않게 (기본 300) |

- **과적합 방지**: 매 에폭 val로 평가해 mAP50-95가 가장 높은 에폭을 `best.pt`로 남기고, 얼리 스톱으로 개선이 멈추면 끝냅니다.
  실제로 47 에폭에서 멈췄고 37 에폭의 가중치를 채택했습니다.
- **현황판**: 에폭이 끝날 때마다 진행률, 남은 시간, 학습률, 손실, 검증 지표, 얼리 스톱 카운터를 출력합니다.
- **중단 후 재개**: 에폭마다 `last.pt`와 백업 `last_backup.pt`를 저장합니다. 같은 명령을 다시 실행하면 가장 최근 학습을
  다음 에폭부터 이어갑니다. `last.pt`가 저장 중 깨졌으면 백업으로 이어갑니다 (최대 1 에폭 손실).
  이어서 할 학습이 있는데 `--epochs` 같은 옵션을 주면, 옵션이 조용히 무시되지 않도록 멈추고 안내합니다.

## 평가 방식

mAP는 박스 위치까지 따지지만, 사람을 세는 데는 "멀리 있는 사람을 몇 % 찾았나"가 더 중요합니다.
그래서 `scripts/evaluate_count.py`로 **사람 크기(=거리) 구간별 recall**을 따로 잽니다. 설정은 `configs/eval.yaml`입니다.

- **정답 매칭**: 검출 박스와 GT의 **visible box**가 IoU 0.5 이상이면 찾은 것으로 봅니다.
  COCO 모델은 보이는 부분만 박스를 치기 때문에, 기본 모델과 공정하게 비교하려고 visible box를 씁니다.
  신뢰도 높은 검출부터 GT 하나에 하나씩만 배정합니다.
- **거리 구간**: GT **full box 높이**로 나눕니다. 가려져도 full box 높이는 거리를 반영하기 때문입니다.
  높이는 모델 입력 해상도 기준 px로 환산하므로 imgsz마다 따로 잽니다.
- **가시율 기준**: CrowdHuman은 가림이 심해서 전체로 재면 거리 효과와 가림 효과가 섞입니다.
  그래서 몸의 75% 이상이 보이는 사람으로 따로 곡선을 구하고, 전체 곡선도 함께 기록합니다.
- **최소 인식 크기**: 가장 큰 구간부터 내려오며 recall이 0.7 이상으로 이어지는 마지막 구간의 하한입니다.
  표본이 50명 미만인 구간은 판정에서 뺍니다. CCTV에 적용할 때는 이 크기보다 작게 보이는 영역을 집계에서 빼는 기준으로 쓸 수 있습니다.

## 실행 방법

Python 3.10~3.12, [uv](https://docs.astral.sh/uv/)를 씁니다. torch는 CUDA 12.6 빌드를 받습니다.

```bash
uv sync --group dev
uv run pytest -q
```

1. [CrowdHuman](https://www.crowdhuman.org/)에서 train/val 이미지와 `annotation_train.odgt`, `annotation_val.odgt`를 받아
   `datasets/crowdhuman/`에 둡니다 (이미지는 하위 폴더 어디에 있어도 됩니다).
2. 변환
   ```bash
   uv run python scripts/convert_crowdhuman.py --src datasets/crowdhuman
   ```
3. 학습 (결과: `runs/YolPe/weights/best.pt`)
   ```bash
   uv run python scripts/train.py                             # 새 학습, 또는 중단된 학습 이어서
   uv run python scripts/train.py --new                       # 처음부터 다시
   uv run python scripts/train.py --epochs 1 --fraction 0.1  # 짧게 시간만 재기
   ```
4. 거리별 평가 (결과: `runs/eval/recall_<모델>_<imgsz>.json`)
   ```bash
   uv run python scripts/evaluate_count.py --src datasets/crowdhuman --model yolo26s.pt              # 기본 모델
   uv run python scripts/evaluate_count.py --src datasets/crowdhuman --model runs/YolPe/weights/best.pt
   ```

## 한계

- **실제 CCTV 영상으로는 아직 평가하지 않았습니다.** CrowdHuman은 사람 눈높이에서 찍은 거리 사진이 많아서,
  위에서 내려다보는 CCTV 시점과는 다릅니다. 실제 적용 전에 해당 카메라 영상으로 다시 재야 합니다.
- 위 수치는 best epoch를 고른 val 세트에서 잰 값이라 실제보다 조금 좋게 나왔을 수 있습니다.
- 많이 가려진 사람은 여전히 놓칩니다. 가려진 사람까지 포함한 전체 recall은 0.686입니다.
- 16px보다 작은 사람은 거의 못 찾습니다.
- 제외 영역을 회색으로 칠했기 때문에 포스터 속 인물, 마네킹, 바닥 반사를 사람으로 볼 수 있습니다. 오탐률은 재지 않았습니다.

## 라이선스

- 코드와 가중치: **AGPL-3.0** (Ultralytics YOLO의 라이선스를 따름)
- CrowdHuman은 **비상업 연구 목적**으로만 쓸 수 있습니다. 데이터와 학습한 가중치를 쓸 때 CrowdHuman 이용 조건을 확인하세요.

## 인용

- Shao et al., "CrowdHuman: A Benchmark for Detecting Human in a Crowd", arXiv:1805.00123, 2018
- Ultralytics YOLO26: https://github.com/ultralytics/ultralytics
