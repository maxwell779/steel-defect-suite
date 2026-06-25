# PRD — Steel Defect Inspection Suite (Severstal)

> 강판 표면결함을 **픽셀 세그멘테이션**으로 검출·분류·위치화하는 end-to-end 포트폴리오.
> 코드(`steel-defect-suite`) + 데모 웹(React) + FastAPI LIVE 추론.
> 기존 KDT 팀의 *패치분류*를 **실제 대회 과제(mean-Dice 세그멘테이션)** 로 정면 돌파.
> *기준일: 2026-06-23. 상세 근거는 [STRATEGY.md](STRATEGY.md).*

## 1. 비전 / 대상
- **한 줄**: "강판 표면에서 결함이 **있는지(분류)** → **무슨 종류(멀티라벨)** → **정확히 어디(픽셀 마스크)** 인지"를 누수 없이, 빈마스크 FP를 억제하며 푸는 검수 콘솔.
- **대상 독자**: 제조 비전·ML 채용 담당. 차별점 = "남들이 비껴간 **대회의 본질(세그멘테이션·empty FP·누수통제)** 을 정조준".

## 2. 목표 / 비목표 / 성공 기준
**목표**
- 실제 Kaggle 과제(RLE 마스크, mean-Dice)를 **leak-free**로 풀어 상위권(0.90) 근접
- 2-stage(분류 게이트 → 세그)로 empty FP 억제 — 대회 점수의 핵심
- **누수 폭로**: 패치 patch-split vs image-split 점수차를 정량 제시(재현·투명성 차별점)
- 보조로 무라벨 이상탐지(ReconPatch) — 그들 AE 0.70 추월
- React 콘솔 + FastAPI LIVE 추론

**비목표**
- ❌ 리더보드 1위 경쟁(상위권 근접+엄정한 평가로 충분)
- ❌ 실시간 라인 연동, 분산학습
- ❌ 합성 결함 생성(보류, copy-paste 증강만)

**성공 기준 (DoD)**
- Stage1 분류 게이트: per-class AUROC·F1 + empty 억제율 보고
- Stage2 세그: **mean Dice ≥ 0.88**, per-class Dice 표, min-size 마스크 정제 ablation
- 누수폭로: patch-split vs image-split Δ 수치
- Stage3 이상탐지: AUROC > 0.70(그들 AE 대비)
- 웹: 강판 업로드 → 4색 마스크 오버레이 + per-class Dice + 비용절감 추정 LIVE
- README 재현법 + 결과표(negative 포함)

## 3. 아키텍처 — 스테이지
| 스테이지 | 질문 | 데이터 | 핵심 기법 | 평가 |
|---|---|---|---|---|
| **0. 베이스라인/누수폭로** | 다른 팀 재현 | 패치 | LGB·ResNet-18, patch-split vs image-split | 누수 Δ 정량 |
| **1. 분류 게이트** | 결함 유무/종류 | 원본 이미지 | EffNet-B3~5 / SE-ResNeXt50 멀티라벨 | per-class AUROC·F1, empty 억제율 |
| **2. 세그멘테이션 ★** | 어디(픽셀) | 원본 + RLE 마스크 | smp FPN/UNet++ × 멀티인코더 앙상블, Lovász/BCE-Dice, TTA, min-size 마스크 정제 | **mean Dice**, per-class Dice |
| **3. 이상탐지(보조)** | 무라벨 가능? | 정상만 | ReconPatch/PatchCore | AUROC |

## 4. 데이터 & 라이선스
> 데이터는 레포 미포함(`.gitignore data/`). 출처에서 직접 받아 `data/`에 둠.

| 데이터 | 출처 | 비고 |
|---|---|---|
| Severstal Steel Defect Detection | Kaggle Competition | 12,568장 256×1600 그레이, RLE 마스크, 결함 4종, 멀티라벨, 빈마스크 다수 |

- 평가는 **(이미지×클래스) mean Dice**: 빈 GT에 예측=0, 미예측=1 → FP 가혹. 자체 GroupKFold val로 산출.

## 5. 데모 웹 (React) — "Steel Inspection Console"
화면 구성(wafer-defect-suite 콘솔 패턴 계승):
1. **통합 콘솔**: KPI(검사수/결함율/격리대기/양품률) + 비용절감 추정 + 결함 큐 + 액션.
2. **Segmentation**: 강판 업로드/선택 → **4색 픽셀 마스크 오버레이**(canvas) + per-class Dice + 신뢰도, threshold/min-size 슬라이더 LIVE.
3. **Classification Gate**: 멀티라벨 확률 막대 + empty 억제 동작 시각화.
4. **Anomaly(보조)**: ReconPatch heatmap.
5. **Experiments**: 누수폭로 차트(patch vs image), 손실 비교(Lovász/BCE-Dice), 인코더/앙상블 gain, per-class Dice, 마스크 정제 ablation, "왜 패치분류가 아니라 세그인가".

백엔드 FastAPI(smp 모델 `*.pt` 추론) 또는 정적 JSON 폴백. 다크모드·CSV/PDF 내보내기.

## 6. 기술 스택 / 구조
- 학습: PyTorch + **segmentation_models.pytorch(smp)** + Albumentations + timm 인코더
- 평가: 자체 Dice(이미지×클래스), GroupKFold(이미지단위)
- 백엔드: FastAPI + `*.pt` 추론 / 프론트: React+Vite, canvas 마스크 오버레이, Recharts
- 구조: `src/`(stage0~3) · `web/` · `experiments/`(git제외) · `docs/` · `data/`(git제외)
- 재현성: seed 고정, `requirements.txt`, image-level fold, threshold val-only

## 7. 산출물
- 학습 모델: 분류 게이트, 세그 앙상블, ReconPatch (`experiments/*/best.pt`, git제외)
- 결과: per-class Dice·누수 Δ·ablation 표, `RESULTS.md`
- 웹 데모(`web/`) + 스크린샷
- 문서: README(재현·결과·출처), 본 PRD, STRATEGY, RESULTS

## 8. 로드맵
1. M0 EDA + image-level fold + RLE 파싱
2. M1 베이스라인 + 누수폭로 + 단일 UNet Dice
3. M2 세그 고도화(모델/손실/마스크 정제/threshold)
4. M3 2-stage + 앙상블 + TTA + pseudo-label → 최종 Dice
5. M4 이상탐지 보조 + negative 정리
6. M5 웹 UI/UX + README/RESULTS + 배포(Pages/Docker)

## 9. 평가 원칙 (busbar/wafer 계승)
- **leak-free**: 이미지단위 GroupKFold, threshold val-only
- **per-class·불균형**: per-class Dice·AUROC, empty FP율, accuracy 단일지표 금지
- **재현·투명성**: 누수 점수도 그대로 폭로, negative(효과없음) 그대로 보고

## 10. 리스크 & 대응
| 리스크 | 대응 |
|---|---|
| B5 풀해상도 메모리 | AMP·grad checkpoint·타일링 |
| test 라벨 비공개 | GroupKFold 자체 Dice, 리더보드는 참고 |
| pseudo-label 누수 | val 격리, pseudo는 train fold만 |
| "전처리 빈약" 인상 | 그들 CLAHE/Bilateral을 증강으로 흡수·계승 명시 |

## 11. 다른 팀 대비 우위 요약
| 항목 | Vision-Q TEAM 1 | 본 프로젝트 |
|---|---|---|
| 과제 | 패치 분류(우회) | **픽셀 세그멘테이션(본질)** |
| 지표 | accuracy 86% | **mean Dice + per-class** |
| 누수 | 의심(patch-split, NB08≠NB12) | **이미지단위 + 누수 폭로** |
| 라벨 | 단일 4종(미검증) | 멀티라벨 + 마스크 |
| 모델 | ResNet-18 / AE 0.70 | EffNet-B5/SE-ResNeXt50 앙상블 + ReconPatch |
| FP | 미고려 | **empty FP 억제(대회 핵심)** |
| 배포 | Streamlit | React 콘솔 + FastAPI LIVE + Pages/Docker |
