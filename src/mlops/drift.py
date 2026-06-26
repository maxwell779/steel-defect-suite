"""데이터 드리프트 모니터 — train vs test 이미지 분포(PSI/KS). steel 이미지용.
이미지 레벨 통계(평균밝기·대비·암부비율) 분포 변화 → 촬영조건 드리프트 감지.
*주의: 진짜 결함 분포 변화 vs 촬영조건 드리프트 구분 한계.* 출처: fiddler PSI, deepchecks KS.

사용: python -m src.mlops.drift
"""
from __future__ import annotations
import json, os, glob
import numpy as np
from PIL import Image
from scipy.stats import ks_2samp
from src.config import ROOT, EXP

OUT = os.path.join(EXP, "mlops"); os.makedirs(OUT, exist_ok=True)
DATA = os.path.join(ROOT, "data")


def stats(paths, n=900, seed=42):
    rng = np.random.default_rng(seed)
    paths = list(paths)
    if len(paths) > n:
        paths = [paths[i] for i in rng.choice(len(paths), n, replace=False)]
    rows = []
    for p in paths:
        a = np.asarray(Image.open(p).convert("L"), np.float32)
        rows.append([a.mean(), a.std(), float((a < 40).mean()), np.percentile(a, 95)])
    return np.array(rows)  # (N,4): mean, std, dark_ratio, p95


def psi(ref, cur, bins=10):
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3: return 0.0
    e = np.clip(np.histogram(ref, edges)[0] / len(ref), 1e-4, None)
    a = np.clip(np.histogram(cur, edges)[0] / len(cur), 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def run():
    tr = stats(glob.glob(os.path.join(DATA, "train_images", "*.jpg")))
    te = stats(glob.glob(os.path.join(DATA, "test_images", "*.jpg")))
    feats = ["mean_brightness", "contrast_std", "dark_ratio", "p95_bright"]
    rows = []
    for i, f in enumerate(feats):
        p = psi(tr[:, i], te[:, i]); ks = ks_2samp(tr[:, i], te[:, i])
        rows.append({"feature": f, "psi": round(p, 4), "ks_stat": round(float(ks.statistic), 4),
                     "ks_p": round(float(ks.pvalue), 4),
                     "level": "alert" if p > 0.25 else ("watch" if p > 0.1 else "stable")})
    share = float(np.mean([r["psi"] > 0.25 for r in rows]))
    out = {"n_train": len(tr), "n_test": len(te), "dataset_drift": share >= 0.5,
           "drift_share": round(share, 3), "features": rows}
    json.dump(out, open(os.path.join(OUT, "drift.json"), "w"), ensure_ascii=False, indent=2)
    for r in rows:
        print(f"[drift] {r['feature']:16s} PSI={r['psi']:.3f} KS={r['ks_stat']:.3f} → {r['level']}", flush=True)
    print(f"dataset_drift={out['dataset_drift']} (share {share:.2f})", flush=True)


if __name__ == "__main__":
    run()
