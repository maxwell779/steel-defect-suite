#!/usr/bin/env bash
# ① OOF용 분류기 fold1~4 학습 (fold0은 이미 존재). 2병렬.
cd /g/steel-defect-suite
export PYTHONIOENCODING=utf-8
PY=/g/anaconda3/python.exe
C="--encoder efficientnet_b3 --epochs 12 --bs 16 --workers 5"
echo "[clf] $(date) — fold1,2 시작"
$PY -m src.train_clf --fold 1 $C > logs/clf_fold1.log 2>&1 &
P=$!
$PY -m src.train_clf --fold 2 $C > logs/clf_fold2.log 2>&1 &
wait $P; wait
echo "[clf] $(date) — fold3,4 시작"
$PY -m src.train_clf --fold 3 $C > logs/clf_fold3.log 2>&1 &
P=$!
$PY -m src.train_clf --fold 4 $C > logs/clf_fold4.log 2>&1 &
wait $P; wait
echo "[clf] $(date) — CLF FOLDS DONE"
