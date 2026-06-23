# Steel Defect Inspection Suite — Severstal

> 강판 표면결함을 **픽셀 세그멘테이션**으로 검출·분류·위치화하는 end-to-end 포트폴리오.
> [Severstal Steel Defect Detection](https://www.kaggle.com/c/severstal-steel-defect-detection) (Kaggle) 데이터.
> 기존 KDT 팀의 *패치분류* 우회를 **실제 대회 과제(mean-Dice 세그멘테이션)** 로 정면 돌파 — leak-free·per-class·정직보고.

## 핵심 차별점
- **본질 정조준**: 패치 분류가 아니라 **픽셀 마스크 세그멘테이션 + 대회 Dice**.
- **누수 통제**: 이미지 단위 5-fold(인접 패치 누수 차단), threshold val-only.
- **empty FP 억제**: 점수의 **86%가 빈 마스크**(EDA §4) → 분류 게이트로 정상 억제.
- **서버 스케일**: A100로 큰 백본(EffNet-B5)·풀해상도·앙상블·TTA·pseudo-label.

## 문서
- [STRATEGY.md](docs/STRATEGY.md) — 다른 팀 분석 · 대회 SOTA 조사 · 고도화 전략
- [PRD.md](docs/PRD.md) — 비전 · 성공기준 · 아키텍처 · 웹 5화면 · 로드맵
- [EDA.md](docs/EDA.md) — 상세 EDA (클래스/면적/멀티라벨/공간분포/밝기누수/fold)

## EDA 요약
| 항목 | 값 |
|---|---|
| train 이미지 | 12,568 (결함 53% / 정상 47%) |
| 클래스 불균형 | C3 73% vs C2 3% (≈21:1) |
| 멀티라벨 | 427장 2종↑ 동시 |
| 빈 마스크 비율 | (이미지×클래스) 50,272쌍 중 **85.9% 빈 마스크** |

![overview](docs/images/eda_overview.png)
![class](docs/images/eda_class_dist.png)

## 파이프라인 (계획)
| 스테이지 | 내용 |
|---|---|
| 0 베이스라인/누수폭로 | 패치분류 재현(patch-split vs image-split Δ 폭로) |
| 1 분류 게이트 | EffNet-B3~5 멀티라벨 → 빈 이미지 억제 |
| 2 세그멘테이션 ★ | smp FPN/UNet++ × {seresnext50, effnet-b5} 앙상블 + Lovász/BCE-Dice + TTA + min-size 후처리 |
| 3 이상탐지(보조) | ReconPatch/PatchCore (무라벨) |

## 재현
```bash
# 1) 데이터: Kaggle에서 받아 압축 해제 → data/  (train.csv, train_images/, test_images/)
# 2) 상세 EDA + leak-free fold 생성
python -m src.eda        # -> docs/EDA.md, data/folds.csv, docs/images/*.png
```

## 데이터 & 출처
- [Severstal Steel Defect Detection](https://www.kaggle.com/c/severstal-steel-defect-detection) — 12,568장 256×1600 그레이, RLE 마스크, 결함 4종.
- 데이터는 레포 미포함(`.gitignore data/`). 출처에서 직접 받아 `data/`에 둘 것.

## 평가 원칙
- **leak-free**: 이미지 단위 fold, threshold val-only
- **per-class·불균형**: per-class Dice·AUROC, empty FP율, accuracy 단일지표 금지
- **정직성**: 누수 점수도 그대로 폭로, negative 결과 그대로 보고
