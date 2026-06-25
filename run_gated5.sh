#!/usr/bin/env bash
cd /g/steel-defect-suite
export PYTHONIOENCODING=utf-8
/g/anaconda3/python.exe -m src.gated_eval --tta --limit 1500 --workers 5 > logs/gated5.log 2>&1
echo "GATED5 DONE rc=$?" >> logs/gated5.log
