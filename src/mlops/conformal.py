"""Conformal 예측 — 분류 게이트(멀티라벨) 결함존재에 분포가정 없는 커버리지 보장.
fold0 val(라벨 있음)을 cal/eval 반분 → 클래스별 conformal 임계 → eval 커버리지 측정.
nonconformity = (1-p) if 결함있음 else p. q=quantile(cal,1-α) → 보장 커버리지.
세그 test는 무라벨이라 conformal은 게이트(분류) 헤드에 적용(turbofan conformal 포팅).

사용: STEEL_DEVICE=cpu python -m src.mlops.conformal --alpha 0.1
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import pandas as pd
import torch
from src import config, data
from src.config import EXP, ROOT

OUT = os.path.join(EXP, "mlops"); os.makedirs(OUT, exist_ok=True)
H, W = 256, 1600
MEAN = np.array([0.344, 0.344, 0.344], np.float32); STD = np.array([0.18, 0.18, 0.18], np.float32)


def clf_predict(ids, ckpt, device, bs=16):
    import timm
    from PIL import Image
    m = timm.create_model("efficientnet_b3", pretrained=False, num_classes=4).to(device).eval()
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    img_dir = os.path.join(ROOT, "data", "train_images")
    out = []
    with torch.no_grad():
        for i in range(0, len(ids), bs):
            batch = []
            for iid in ids[i:i + bs]:
                a = Image.open(os.path.join(img_dir, iid)).convert("RGB").resize((W, H))
                a = (np.asarray(a, np.float32) / 255 - MEAN) / STD
                batch.append(a.transpose(2, 0, 1))
            x = torch.tensor(np.stack(batch)).to(device)
            out.append(torch.sigmoid(m(x)).cpu().numpy())
    return np.concatenate(out)


def run(alpha=0.1, fold=0, seed=42):
    dev = os.environ.get("STEEL_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    _, va_ids = data.split_fold(fold)
    va_ids = list(va_ids)
    # GT 멀티라벨(클래스 존재 여부)
    df = pd.read_csv(os.path.join(ROOT, "data", "train.csv"))
    present = {}
    for iid, c in zip(df["ImageId"], df["ClassId"]):
        present.setdefault(iid, set()).add(int(c))
    Y = np.array([[1 if k in present.get(iid, set()) else 0 for k in (1, 2, 3, 4)] for iid in va_ids])
    P = clf_predict(va_ids, os.path.join(EXP, f"stage1_clf_efficientnet_b3_f{fold}", "best.pt"), dev)
    # cal/eval 반분
    rng = np.random.default_rng(seed); idx = rng.permutation(len(va_ids)); half = len(idx) // 2
    cal, ev = idx[:half], idx[half:]
    res = {"alpha": alpha, "target_coverage": round(1 - alpha, 2), "n_cal": int(half), "n_eval": int(len(ev)), "per_class": []}
    covs = []
    for k in range(4):
        nc = np.where(Y[cal, k] == 1, 1 - P[cal, k], P[cal, k])
        n = len(nc); q = float(np.quantile(nc, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)))
        # eval: 클래스 포함여부 집합 — 결함있다고 보장하는 구간 커버리지
        cover = np.where(Y[ev, k] == 1, (1 - P[ev, k]) <= q, P[ev, k] <= q)
        cov = float(cover.mean()); covs.append(cov)
        res["per_class"].append({"class": f"C{k+1}", "q": round(q, 3), "empirical_coverage": round(cov, 3)})
    res["mean_coverage"] = round(float(np.mean(covs)), 3)
    json.dump(res, open(os.path.join(OUT, "conformal_gate.json"), "w"), indent=2)
    print(f"[conformal-gate] 목표 {1-alpha:.0%} → 클래스별 커버리지 " +
          " ".join(f"C{k+1}={res['per_class'][k]['empirical_coverage']:.0%}" for k in range(4)) +
          f" (평균 {res['mean_coverage']:.1%})", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--alpha", type=float, default=0.1); ap.add_argument("--fold", type=int, default=0)
    a = ap.parse_args(); run(a.alpha, a.fold)
