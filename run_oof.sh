#!/usr/bin/env bash
cd /g/steel-defect-suite
export PYTHONIOENCODING=utf-8
/g/anaconda3/python.exe -m src.oof_gated_eval --tta --workers 5 > logs/oof_full.log 2>&1
echo "OOF SCRIPT DONE rc=$?" >> logs/oof_full.log
