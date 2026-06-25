"""게이트 결합 평가 (M5) — 분류 게이트 × 세그 앙상블 = gated mean Dice.

흐름: 4모델 앙상블(+TTA) → 3-임계 후처리 → **분류기로 (이미지,클래스) 없음 판정 시 마스크 끔**.
- 게이트 임계는 클래스별로 독립 최적화(클래스c 게이트는 클래스c Dice에만 영향 → greedy 최적).
- ungated(0.9534 재현) vs gated 비교 + per-class.

실행: python -m src.gated_eval --tta --limit 1500
"""
import os, json, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from src import config, data
from src.metrics import dice_pair
from src.postprocess import MEMBERS, ARCH, predict_probs, gts_of, apply_postproc, \
    mean_dice_from_pred, per_class_dice
from src.train_clf import SteelClfDataset, clf_tfms
import timm


@torch.no_grad()
def clf_probs(ids, ann, encoder, ckpt, device, bs, workers):
    ld = DataLoader(SteelClfDataset(ids, ann, clf_tfms(False)),
                    batch_size=bs, shuffle=False, num_workers=workers)
    m = timm.create_model(encoder, pretrained=False, num_classes=config.N_CLASSES).to(device).eval()
    m.load_state_dict(torch.load(ckpt, map_location=device))
    P = []
    for x, _, _ in ld:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=True):
            P.append(torch.sigmoid(m(x)).float().cpu().numpy())
    return np.concatenate(P)  # (N,C)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=1500)
    ap.add_argument("--clf_encoder", default="efficientnet_b3")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ann = data.load_annotations()
    _, va_ids = data.split_fold(args.fold)
    if args.limit:
        va_ids = va_ids[:args.limit]
    va = DataLoader(data.SteelSegDataset(va_ids, ann, data.build_tfms(False)),
                    batch_size=args.bs, shuffle=False, num_workers=args.workers)
    print(f"[data] val {len(va_ids)}", flush=True)
    gts = gts_of(va)

    # 1) 세그 앙상블(+TTA)
    members = [(a, e, d) for a, e, d in MEMBERS if os.path.exists(os.path.join(config.EXP, d, "best.pt"))]
    ens = None
    for a, e, d in members:
        m = ARCH[a](encoder_name=e, encoder_weights=None, in_channels=3, classes=4).to(device).eval()
        m.load_state_dict(torch.load(os.path.join(config.EXP, d, "best.pt"), map_location=device))
        p = predict_probs(m, va, device, args.tta).astype(np.float32)
        ens = p if ens is None else ens + p
        del m; torch.cuda.empty_cache()
    ens /= len(members)
    print(f"[seg] 앙상블 {len(members)}모델{' +TTA' if args.tta else ''}", flush=True)

    # 2) 후처리 그리드 → ungated best
    best = (mean_dice_from_pred(ens >= 0.5, gts), (0.5, 0.0, 0))
    for mp in [0.4, 0.5, 0.6]:
        for xp in [0.5, 0.6, 0.7]:
            for ma in [0, 600, 1200, 2000]:
                md = mean_dice_from_pred(apply_postproc(ens, mp, xp, ma), gts)
                if md > best[0]:
                    best = (md, (mp, xp, ma))
    ungated, (mp, xp, ma) = best
    pred = apply_postproc(ens, mp, xp, ma)          # (N,C,H,W) bool
    pc_ungated = per_class_dice(pred, gts)
    print(f"[ungated] mean Dice {ungated:.4f} (min_prob={mp},max_prob={xp},min_area={ma}) "
          f"per-class={[round(x,3) for x in pc_ungated]}", flush=True)

    # 3) 분류 게이트
    ck = os.path.join(config.EXP, f"stage1_clf_{args.clf_encoder}_f{args.fold}", "best.pt")
    cp = clf_probs(va_ids, ann, args.clf_encoder, ck, device, args.bs, args.workers)  # (N,C)
    N, C = pred.shape[:2]
    # 게이트 상태별 dice 사전계산: 유지(seg) vs 억제(빈마스크)
    kept = np.array([[dice_pair(pred[n, c], gts[n, c]) for c in range(C)] for n in range(N)])  # (N,C)
    supp = np.array([[1.0 if gts[n, c].sum() == 0 else 0.0 for c in range(C)] for n in range(N)])
    grid = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 0.95]
    gate_thr, pc_gated = [], []
    for c in range(C):
        bestc = (kept[:, c].mean(), 0.0)  # thr=0 → 게이트 없음(전부 유지)
        for t in grid:
            keep = cp[:, c] >= t
            dc = np.where(keep, kept[:, c], supp[:, c]).mean()
            if dc > bestc[0]:
                bestc = (dc, t)
        pc_gated.append(float(bestc[0])); gate_thr.append(float(bestc[1]))
    gated = float(np.mean(pc_gated))
    print(f"[gated]   mean Dice {gated:.4f}  게이트임계(C1~C4)={[round(t,3) for t in gate_thr]}", flush=True)
    print(f"          per-class={[round(x,3) for x in pc_gated]}  (Δ {gated-ungated:+.4f})", flush=True)

    out = {"members": [d for _, _, d in members], "tta": args.tta, "n_val": len(va_ids),
           "ungated_dice": ungated, "ungated_per_class": pc_ungated,
           "postproc": {"min_prob": mp, "max_prob": xp, "min_area": ma},
           "clf_encoder": args.clf_encoder, "gate_thr": gate_thr,
           "gated_dice": gated, "gated_per_class": pc_gated, "delta": gated - ungated}
    op = os.path.join(config.ROOT, "docs", "overnight", "GATED_RESULTS.json")
    os.makedirs(os.path.dirname(op), exist_ok=True)
    with open(op, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("saved ->", op, flush=True)


if __name__ == "__main__":
    main()
