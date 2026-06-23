# RESULTS — Steel Defect Inspection Suite (Severstal)

> 실제 결과 종합. 모든 평가는 **이미지 단위 fold(누수 차단)**. 상세 EDA는 [EDA.md](EDA.md).
> *기준일: 2026-06-23. 현재 fold0 단일모델 기준(앙상블·다fold는 진행 예정).*

## 요약
| 스테이지 | 지표 | 결과 |
|---|---|---|
| 0 누수 폭로 | 패치분류 patch-split vs image-split | acc Δ **+1.25%p**, AUC Δ +0.009 (누수 실측) |
| 2 세그멘테이션 ★ | **대회식 mean Dice** (UNet/se_resnext50, fold0) | **0.9317** → min-size 후처리 **0.9343** |

> 참고: Severstal 공개 리더보드 상위권은 private test 기준 ~0.90–0.918. 본 수치는 **train 이미지 단위 held-out fold** 기준이라 리더보드와 직접 비교는 아니지만, **단일모델·단일fold 베이스라인으로 매우 강함**.

## Stage 0 — 패치분류 누수 폭로 (정직성 차별점)
다른 팀(KDT Vision-Q)은 세그멘테이션을 **256×256 패치 이진분류**로 우회했고, 50% overlap 패치를 분할하면 누수 위험이 있다(그들 스스로 "NB08 90.9% vs NB12 93.3% 분할차" 보고). 이를 동일 분류기로 정량화:

| 분할 | AUC | acc | f1 |
|---|---|---|---|
| patch-split (누수) | 0.9821 | 0.9514 | 0.8800 |
| image-split (정상) | 0.9735 | 0.9389 | 0.8510 |
| **Δ 누수 인플레** | +0.009 | **+1.25%p** | +0.029 |

- 누수는 **실재하지만 modest**(이진 패치 수준). → 핵심 문제는 누수 크기가 아니라 **그들이 과제 본질(픽셀 Dice + empty-FP 억제)을 통째로 비껴갔다**는 점.
- 코드: `src/patch_leak.py` (4000장·4ep, resnet18).

## Stage 2 — 세그멘테이션 베이스라인 ★
- 모델: **smp UNet + se_resnext50_32x4d**(imagenet), in=3 / out=4(멀티라벨 마스크), 입력 256×1600 풀해상도.
- 학습: BCE+Dice(0.6/0.4), AdamW lr 3e-4, cosine 12ep, AMP, bs16, A100.
- 평가: **대회식 (이미지×클래스) mean Dice**, 이미지 단위 fold0(train 10054 / val 2514).

| | mean Dice | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|
| best (no postproc) | **0.9317** | 0.928 | 0.981 | **0.835** | 0.983 |
| + min-size 후처리(800) | **0.9343** | | | | |

- **C3(대형/스크래치)가 천장**(0.835) — 가장 흔하고 면적 변동이 큰 클래스. C2(희귀)는 0.981로 오히려 높음(빈 마스크 정확 억제 + 형태 단순).
- min-size 후처리로 작은 거짓 마스크 제거 → +0.0026. (EDA §5b의 클래스별 p5 면적 가이드 근거)
- 코드: `src/train_seg.py`, `src/metrics.py`(대회 Dice), `src/data.py`(RLE↔mask).

## 다음 (계획)
- 분류 게이트(EffNet) → empty-FP 추가 억제, 5-fold 앙상블, FPN/UNet++ 다인코더, TTA, pseudo-label.
- per-fold·앙상블 Dice → 본 표 갱신.
