#!/usr/bin/env bash
# M5 watcher — 현재 체인(run_chain.sh)의 M4 완료(ALL DONE) 감지 후 자동 실행:
#  (1) 승자(DeepLabV3+/effb3, bce_lovasz) 5-fold (fold1~4; fold0은 이미 존재) → fold 앙상블
#  (2) 분류 게이트(멀티라벨 EfficientNet-B3) — empty-FP 억제용
cd /g/steel-defect-suite
export PYTHONIOENCODING=utf-8
PY=/g/anaconda3/python.exe
SEG="--arch deeplabv3plus --encoder efficientnet-b3 --loss bce_lovasz --oversample --posweight 4,8,1,2 --epochs 16 --bs 16 --workers 5 --tag _5fold"

echo "[m5] $(date) — M4(ALL DONE) 대기..."
until grep -q "ALL DONE" logs/chain.log 2>/dev/null; do sleep 120; done
echo "[m5] $(date) — M4 완료 감지 → M5 시작"

# (1) 승자 5-fold: fold1&2 병렬 → fold3&4 병렬
$PY -m src.train_seg --fold 1 $SEG > logs/m5_seg_fold1.log 2>&1 &
P=$!
$PY -m src.train_seg --fold 2 $SEG > logs/m5_seg_fold2.log 2>&1 &
wait $P; wait
echo "[m5] $(date) — fold 1,2 완료"
$PY -m src.train_seg --fold 3 $SEG > logs/m5_seg_fold3.log 2>&1 &
P=$!
$PY -m src.train_seg --fold 4 $SEG > logs/m5_seg_fold4.log 2>&1 &
wait $P; wait
echo "[m5] $(date) — fold 3,4 완료 → 5-fold 학습 끝"

# (2) 분류 게이트(fold0 전체)
echo "[m5] $(date) — 분류 게이트 학습 시작"
$PY -m src.train_clf --fold 0 --encoder efficientnet_b3 --epochs 12 --bs 16 --workers 5 > logs/m5_clf.log 2>&1
echo "[m5] $(date) — 분류 게이트 완료. M5 DONE"
