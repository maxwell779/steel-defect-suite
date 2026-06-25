#!/usr/bin/env bash
# ② Pseudo-label 2라운드 — 누수 차단(test→train fold0만, val 격리).
cd /g/steel-defect-suite
export PYTHONIOENCODING=utf-8
PY=/g/anaconda3/python.exe
WIN=stage2_seg_deeplabv3plus_efficientnet-b3_f0_sw_f_deeplabv3plus_effb3_x3
R1=stage2_seg_deeplabv3plus_efficientnet-b3_f0_pseudo_r1

echo "[ps] $(date) — R1 generate (teacher=승자)"
$PY -m src.pseudo_label generate --teacher $WIN --clf stage1_clf_efficientnet_b3_f0 \
    --out data/pseudo_r1.csv --limit 3000 --workers 5 > logs/pseudo_gen_r1.log 2>&1
echo "[ps] $(date) — R1 train (student)"
$PY -m src.pseudo_label train --pseudo data/pseudo_r1.csv --tag _pseudo_r1 \
    --epochs 12 --workers 5 > logs/pseudo_train_r1.log 2>&1
echo "[ps] $(date) — R2 generate (teacher=R1 student)"
$PY -m src.pseudo_label generate --teacher $R1 --clf stage1_clf_efficientnet_b3_f0 \
    --out data/pseudo_r2.csv --limit 3000 --workers 5 > logs/pseudo_gen_r2.log 2>&1
echo "[ps] $(date) — R2 train (warm-start from R1)"
$PY -m src.pseudo_label train --pseudo data/pseudo_r2.csv --tag _pseudo_r2 --init $R1 \
    --epochs 12 --workers 5 > logs/pseudo_train_r2.log 2>&1
echo "[ps] $(date) — PSEUDO DONE"
