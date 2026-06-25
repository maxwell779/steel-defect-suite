# Steel Inspection Console (web)

Severstal 검수 데모 — FastAPI 실추론 + 무빌드 SPA(canvas 마스크 오버레이·Chart.js).

## 실행
```bash
pip install fastapi uvicorn python-multipart pillow      # (torch/smp/timm은 학습 환경 공용)
# 데모 샘플 사전계산(최초 1회, CPU 가능)
STEEL_DEVICE=cpu python -m web.precompute_samples
# 서버
STEEL_DEVICE=cpu uvicorn web.server:app --host 127.0.0.1 --port 8010
# 브라우저: http://127.0.0.1:8010
```
- `STEEL_DEVICE=cpu` 권장(학습 중 GPU 충돌 회피). GPU 쓰려면 `STEEL_DEVICE=cuda`.
- 모델 `.pt`가 없으면 **정적 폴백**(사전계산 샘플)으로 동작.

## 구성
- `infer.py` — 공용 추론(승자 DeepLabV3+/effb3 세그 + EffNet-B3 분류 게이트 → 4색 마스크).
- `server.py` — FastAPI: `/`(SPA), `/api/health|samples|experiments`, `POST /api/infer`(업로드 LIVE).
- `precompute_samples.py` — fold0 val 대표 6장 사전 추론 → `static/samples/`.
- `static/` — index.html · app.js · experiments.json · samples/.

## 화면 (PRD §5)
1. **통합 콘솔** — KPI(검사/결함율/격리/양품률/비용절감) + 결함 큐 + 최종 성능.
2. **Segmentation** — 샘플/업로드 → 4색 마스크 오버레이 + per-class Dice/확률/면적, 후처리·게이트 슬라이더 LIVE(업로드).
3. **Classification Gate** — 멀티라벨 확률 + empty 억제(차단/통과) + AUROC·차단율 차트.
4. **Experiments** — 마일스톤 Dice, 레버 이득, per-class(게이트 전/후), 전처리 교차검증, 누수 폭로, 5-fold, 타팀 대비표.
