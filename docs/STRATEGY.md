# 전략 — Severstal Steel Defect Detection 고도화

> 목표: KDT 다른 팀(Vision-Q TEAM 1)의 "철강 품질 실패비용 최소화 검수 시스템"을
> **실제 대회 과제(픽셀 세그멘테이션)** 로 정면 돌파하고, A100 서버와
> wafer/busbar에서 검증된 *leak-free·per-class·정직보고* 방법론으로 추월한다.
> *기준일: 2026-06-23*

---

## 1. 다른 팀(Vision-Q)이 한 것 — 분석

### 1.1 그들의 접근 (PDF 80p 정독)
- **데이터**: Severstal 12,568장(256×1600 그레이). 슬라이딩 윈도우(stride=128, 50% overlap) → 256×256 패치 12장/이미지 = **150,816 패치(정상 120,583 + 결함 30,233)**.
- **전처리(자료의 ~50%)**: 정렬/Auto-deskew(Otsu→contour→minAreaRect) → 에지밀도 ROI(Auto Canny + 4×4 그리드) → CLAHE(clip=2.0) → Bilateral(d=9,σ=75) → Adaptive Sharpen → Opening. 정량비교(ED/LV)로 각 단계 선택 근거 제시. **여기까지는 매우 탄탄.**
- **특징공학**: HOG + LBP + 픽셀통계 + 윤곽선형태 + 에지 = **1,944-D 벡터**, t-SNE 시각화, 상관 2,610쌍.
- **모델(하이브리드 2-stage)**:
  - Stage1 이진(정상 vs 결함): ML LightGBM AUC 0.801/acc 81.6%, DL ResNet-18 FT **acc 93.25%**.
  - Stage2 결함 4종: ML LGB acc 77.7%/F1 0.775, DL ResNet-18 Feature-Extractor **88.6%**(최종 4종 acc 86.0%).
  - 이상탐지: Conv-AE **AUROC 0.7023**, threshold에서 recall 96%.
- **배포**: Streamlit 대시보드.
- **한계 자인**: 클래스 불균형(4:1), 단일 촬영조건, **라벨품질 미검증**, AE 0.70 한계, 설명성 부족.

### 1.2 치명적 결함 — 우리가 이기는 지점
| # | 그들의 문제 | 왜 문제인가 | 우리의 해법 |
|---|---|---|---|
| **A** | **실제 과제를 안 풀었다** | Severstal은 **픽셀 세그멘테이션(RLE 마스크, mean-Dice)** 인데 *패치 분류*로 치환. Dice도, 리더보드 비교도, 위치 마스크도 없음 | smp UNet/FPN로 **진짜 세그멘테이션** + 대회 Dice 산출 → 리더보드(0.90+)와 직접 비교 |
| **B** | **패치 단위 분할 = 누수 의심** | 50% overlap 패치를 patch-level로 split하면 인접 패치가 train/test에 동시 존재 → 점수 부풀림. 그들 스스로 "NB08 90.9% vs NB12 93.3% (샘플링/분할 차이)" 보고 = 누수 적신호 | **이미지(파일) 단위 GroupKFold** 고정. 패치는 학습편의일 뿐, 평가는 원본 이미지 |
| **C** | **멀티라벨을 단일라벨로 뭉갬** | Severstal은 한 이미지에 결함 2종↑ 동시 존재 가능(멀티라벨). 4종 단일분류는 문제정의 오류 | 4채널 멀티라벨 마스크 + per-class Dice |
| **D** | **정확도 중심 평가** | 4:1 불균형에서 accuracy는 무의미. 대회는 **빈 마스크 FP를 가혹하게 처벌**(empty GT에 결함 예측=Dice 0) | empty-image FP율을 1급 지표로. classifier 게이트로 정상이미지 억제 |
| **E** | **ResNet-18 / AE 0.70만** | 현대 백본·세그 모델 미사용. AE AUROC 0.70은 약함 | EfficientNet-B5+/SE-ResNeXt50 인코더, UNet++/FPN, 앙상블·TTA |
| **F** | ROI가 에지밀도 휴리스틱뿐 | 진짜 위치(마스크)가 아님 | 픽셀 마스크 + 오버레이 시각화 |

> **한 줄 요약**: 그들은 *전처리·특징공학은 훌륭*하나 **대회의 본질(세그멘테이션+empty FP 억제+누수통제)** 을 비껴갔다. 우리는 정확히 그 본질을 친다.

---

## 2. 실제 Kaggle 대회 — 조사 결과

### 2.1 과제 정의
- **입력**: 256×1600 그레이 강판 이미지 12,568장(train). 결함 4종(ClassId 1~4).
- **출력**: 클래스별 **픽셀 마스크(RLE)**.
- **평가지표**: **(이미지 × 클래스)별 mean Dice**. 한 이미지당 4개 예측.
  - GT가 빈 마스크인데 **아무것도 예측 안 하면 Dice=1.0**, 한 픽셀이라도 예측하면 **Dice=0.0**.
  - → **거짓양성(FP) 억제가 점수의 핵심.** 대부분 이미지·클래스 쌍이 "빈 마스크"라 정상 억제가 절반 이상의 점수를 좌우.
- **불균형**: Class 3(대형/스크래치)가 압도적 다수, Class 2(가장 희귀)는 수백 장.

### 2.2 상위권 공통 레시피 (조사 종합)
1. **2-stage = 분류 게이트 → 세그멘테이션**: 먼저 classifier로 "결함 있음/클래스별 유무" 판정해 **빈 이미지를 거른 뒤**, 양성으로 판정된 것만 세그. empty FP 폭증을 막는 정석.
2. **세그 모델(smp)**: **FPN / UNet / UNet++**, 인코더 **SE-ResNeXt50_32x4d**(bs 32~36), **EfficientNet-B5**(bs 16~20), inceptionv4 등. 상위 앙상블 = 여러 인코더 평균.
3. **손실**: **Lovász**가 BCE/BCE-Dice보다 public/val에서 "극적으로" 우세. 단, **분류모델과 결합 시엔 BCE+Dice**가 더 좋았다(Lovász가 FP 억제 역할). → 분류게이트 채택 시 **BCE+Dice(0.6·BCE+0.4·(1−Dice))** 기본, Lovász는 게이트 없는 단일모델 부스트용.
4. **후처리**: **작은 마스크 제거(min mask size threshold)** + 구멍 메우기(hole fill). connected-component는 효과 없었음.
5. **TTA**: h/v flip.
6. **의사라벨(pseudo-labeling) 2라운드**: 상위권 대부분 사용, 유의미한 향상.
7. **CV**: 5~10 fold, threshold는 val에서만.
8. **성능 기준선**: 단일 강한 모델 ~0.90 Dice(private), 상위 앙상블 **0.908~0.918**. (다른 팀은 Dice 자체가 없음.)

---

## 3. 우리의 고도화 전략 (A100 서버 활용)

### 3.1 핵심 차별화 3축
1. **본질 정조준**: 패치분류가 아니라 **픽셀 세그멘테이션 + 대회 Dice**. 리더보드와 1:1 비교.
2. **누수통제 + 정직평가**: 이미지단위 GroupKFold, per-class Dice, **empty FP율 명시**, negative 결과 그대로 보고(wafer/busbar 계승).
3. **서버 스케일**: 큰 백본(B5+)·풀해상도(256×1600 또는 큰 타일)·앙상블·TTA·pseudo-label·SWA/EMA·AMP·큰 배치.

### 3.2 파이프라인 (3-스테이지 + 보조)
| 스테이지 | 질문 | 방법 | 지표 |
|---|---|---|---|
| **0. (재현) 베이스라인** | 그들 패치분류 재현 | LGB/ResNet-18 패치분류 (누수 있는 split / 없는 split 둘 다) | **누수가 점수를 얼마나 부풀리는지 정량 폭로** |
| **1. 분류 게이트** | 결함 있나/어느 클래스 | EfficientNet-B3~B5 / SE-ResNeXt50 멀티라벨(4-logit) | per-class AUROC·F1, **empty 정확 억제율** |
| **2. 세그멘테이션 ★** | 어디에 (픽셀) | smp FPN/UNet++ × {seresnext50, effnet-b5} 앙상블 + Lovász/BCE-Dice + TTA + min-size 후처리 | **대회 mean Dice**, per-class Dice |
| **3. 이상탐지(보조)** | 라벨없이 가능? | 우리 강점 **ReconPatch/PatchCore**(정상만 학습) → 무라벨 검출 baseline | AUROC, AE 0.70 대비 우위 |

> Stage0(재현+누수폭로)은 **이 프로젝트의 "정직" 차별점**. "다른 팀 93%는 누수 포함 가능성, 누수 제거 시 XX%"를 수치로 보여주는 게 포트폴리오의 킬러 포인트.

### 3.3 불균형·FP 대응 (대회 점수의 핵심)
- 분류 게이트로 빈 이미지 선제 차단(empty FP의 1차 방어).
- 세그 손실: BCE+Dice(+Lovász), per-class pos_weight.
- 후처리: 클래스별 **min mask size**를 val에서 튜닝(작은 거짓 마스크 제거 = Dice 직접 상승).
- 증강: flip, brightness/contrast, **copy-paste(희귀 Class2 결함 합성)**.
- threshold(분류·픽셀·min-size) 전부 **val에서만** 결정.

### 3.4 A100 활용 계획
- 풀해상도 학습(256×1600) + AMP + 큰 배치, 멀티 인코더 병렬(GPU guard).
- 10-fold 중 일부 fold 동시 학습.
- pseudo-label 2라운드(test/unlabeled 활용).
- SWA/EMA, cosine LR, gradient checkpointing(B5 풀해상도용).

### 3.5 마일스톤
1. **M0 데이터·EDA**: RLE 파싱, 클래스/마스크 면적 분포, 빈마스크 비율, 이미지단위 fold 생성.
2. **M1 베이스라인+누수폭로**: 패치분류 재현(누수/무누수), 단일 UNet(seresnext50) Dice baseline.
3. **M2 세그 고도화**: FPN/UNet++ × 멀티인코더, 손실/후처리/threshold 튜닝 → 단일 best.
4. **M3 2-stage + 앙상블 + TTA + pseudo-label** → 최종 Dice.
5. **M4 이상탐지 보조**(ReconPatch) + negative 정리.
6. **M5 UI/UX**(§ PRD) + README/RESULTS + 배포(Pages/Docker).

### 3.6 성공 기준
- 세그 mean Dice **≥ 0.88**(상위권 0.90 근접), per-class Dice 표.
- **누수폭로**: 패치 patch-split vs image-split 점수차 정량 보고.
- empty FP율 명시 + min-size 후처리 기여도 ablation.
- ReconPatch 이상탐지 AUROC > 그들 AE 0.70.
- 웹 콘솔: 강판 업로드 → 4색 마스크 오버레이 + per-class Dice + 비용절감 추정 LIVE.

---

## 4. 리스크
| 리스크 | 대응 |
|---|---|
| 풀해상도 B5 메모리 | AMP + grad checkpoint + 배치 조정, 안되면 큰 타일(512×512) |
| 대회 test 라벨 비공개 | train을 GroupKFold val로 자체 Dice 산출(리더보드는 참고치) |
| pseudo-label 누수 | val fold 격리 유지, pseudo는 train fold에만 |
| 그들 대비 "전처리 빈약" 인상 | 그들 CLAHE/Bilateral 파이프라인을 **세그 입력 증강으로 흡수**(존중+계승) |

## 5. 데이터 & 출처
- Severstal Steel Defect Detection (Kaggle): https://www.kaggle.com/c/severstal-steel-defect-detection
- smp(segmentation_models.pytorch), Albumentations.
- 참고 솔루션: khornlund, TheoViel, diyago(Insaf Ashrapov) 공개 레포/블로그.
