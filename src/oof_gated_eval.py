"""① 5-fold OOF 게이트 결합 — 누수 없는 다fold 일반화.

각 fold k: 승자(deeplabv3plus/effb3) **fold-k 모델이 자기 val_k만** 예측(OOF, 무누수)
→ 3-임계 후처리 → **fold-k 분류기**로 클래스별 빈마스크 게이트.
임계는 fold0 튜닝값으로 **고정**(per-fold 튜닝 안 함 = 정직한 전이). 전체 train OOF 집계.

실행: python -m src.oof_gated_eval --tta
"""
import os, json, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from src import config, data
from src.metrics import dice_pair
from src.postprocess import ARCH, predict_probs, gts_of, apply_postproc
from src.gated_eval import clf_probs

# fold별 승자 seg 디렉터리
SEG_DIR = {0: "stage2_seg_deeplabv3plus_efficientnet-b3_f0_sw_f_deeplabv3plus_effb3_x3"}
for k in (1, 2, 3, 4):
    SEG_DIR[k] = f"stage2_seg_deeplabv3plus_efficientnet-b3_f{k}_5fold"

POSTPROC = (0.6, 0.7, 600)          # M4 튜닝값 고정
GATE_THR = [0.3, 0.02, 0.5, 0.0]    # fold0 게이트 임계 고정


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="fold당 val 상한(0=전체)")
    ap.add_argument("--clf_encoder", default="efficientnet_b3")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ann = data.load_annotations()
    mp, xp, ma = POSTPROC

    ung_all, gat_all = [], []   # 전체 OOF per-(n,c) dice 누적
    per_fold = {}
    for k in range(5):
        seg_ck = os.path.join(config.EXP, SEG_DIR[k], "best.pt")
        clf_ck = os.path.join(config.EXP, f"stage1_clf_{args.clf_encoder}_f{k}", "best.pt")
        if not (os.path.exists(seg_ck) and os.path.exists(clf_ck)):
            print(f"[fold{k}] skip — 가중치 없음 (seg={os.path.exists(seg_ck)} clf={os.path.exists(clf_ck)})", flush=True)
            continue
        _, va_ids = data.split_fold(k)
        if args.limit:
            va_ids = va_ids[:args.limit]
        va = DataLoader(data.SteelSegDataset(va_ids, ann, data.build_tfms(False)),
                        batch_size=args.bs, shuffle=False, num_workers=args.workers)
        gts = gts_of(va)
        # seg 예측(TTA) + 후처리
        m = ARCH["deeplabv3plus"](encoder_name="efficientnet-b3", encoder_weights=None,
                                  in_channels=3, classes=4).to(device).eval()
        m.load_state_dict(torch.load(seg_ck, map_location=device))
        probs = predict_probs(m, va, device, args.tta).astype(np.float32)
        del m; torch.cuda.empty_cache()
        pred = apply_postproc(probs, mp, xp, ma)            # (N,C,H,W) bool
        # 분류 게이트(fold-k 분류기)
        cp = clf_probs(va_ids, ann, args.clf_encoder, clf_ck, device, args.bs, args.workers)  # (N,C)
        N, C = pred.shape[:2]
        ung = np.array([[dice_pair(pred[n, c], gts[n, c]) for c in range(C)] for n in range(N)])
        supp = np.array([[1.0 if gts[n, c].sum() == 0 else 0.0 for c in range(C)] for n in range(N)])
        gat = np.where(cp >= np.array(GATE_THR)[None, :], ung, supp)
        ung_all.append(ung); gat_all.append(gat)
        per_fold[k] = {"n": int(N), "ungated": float(ung.mean()), "gated": float(gat.mean())}
        print(f"[fold{k}] n={N} ungated={ung.mean():.4f} gated={gat.mean():.4f}", flush=True)

    U = np.concatenate(ung_all); G = np.concatenate(gat_all)   # (총N,4)
    res = {
        "method": "5-fold OOF (승자 deeplabv3plus/effb3, fold별 모델·분류기, 임계 고정)",
        "postproc": {"min_prob": mp, "max_prob": xp, "min_area": ma}, "gate_thr": GATE_THR,
        "n_total": int(U.shape[0]), "folds": per_fold,
        "oof_ungated": float(U.mean()), "oof_gated": float(G.mean()),
        "oof_ungated_per_class": [float(U[:, c].mean()) for c in range(4)],
        "oof_gated_per_class": [float(G[:, c].mean()) for c in range(4)],
        "delta": float(G.mean() - U.mean()),
    }
    print(f"\n=== 5-fold OOF (전체 train {res['n_total']}장, 무누수) ===")
    print(f"  ungated {res['oof_ungated']:.4f}  per-class {[round(x,3) for x in res['oof_ungated_per_class']]}")
    print(f"  gated   {res['oof_gated']:.4f}  per-class {[round(x,3) for x in res['oof_gated_per_class']]}  (Δ {res['delta']:+.4f})")
    op = os.path.join(config.ROOT, "docs", "overnight", "OOF_GATED_RESULTS.json")
    os.makedirs(os.path.dirname(op), exist_ok=True)
    with open(op, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("saved ->", op, flush=True)


if __name__ == "__main__":
    main()
