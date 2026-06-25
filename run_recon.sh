#!/usr/bin/env bash
cd /g/steel-defect-suite
export PYTHONIOENCODING=utf-8
PY=/g/anaconda3/python.exe
{
echo "### layer3 max"
$PY -m src.recon_patch --normals 1200 --val 1500 --layer 3 --score max --per_img 256 --bank 40000 --bs 8
echo "### layer2 max"
$PY -m src.recon_patch --normals 1200 --val 1500 --layer 2 --score max --per_img 256 --bank 40000 --bs 8
echo "### layer3 topk"
$PY -m src.recon_patch --normals 1200 --val 1500 --layer 3 --score topk --per_img 256 --bank 40000 --bs 8
} > logs/recon_cmp.log 2>&1
echo "RECON CMP DONE" >> logs/recon_cmp.log
