"""③ ReconPatch/PatchCore-계열 무라벨 이상탐지 (보조).

정상(결함 없는) 이미지의 패치 특징으로 메모리뱅크 구성 → 테스트 패치의 최근접 거리로 이상 점수.
라벨 없이 동작. **이미지단위 AUROC**로 다른 팀 Conv-AE 0.70 대비 우위 확인.

- backbone: timm resnet18 features(layer2, ImageNet) — 패치 임베딩.
- fit: fold0 train의 정상(empty) 이미지에서 패치 샘플 → 코어셋(랜덤 서브샘플) 뱅크.
- score: val 이미지 패치 → 뱅크 kNN 거리 → 이미지점수=상위 top-k 패치 평균.

실행: python -m src.recon_patch --normals 1200 --val 1500 --bank 30000
"""
import os, json, argparse, time
import numpy as np
from PIL import Image
import torch
import timm

from src import config, data

H, W = config.H, config.W
MEAN = np.array([0.344, 0.344, 0.344], np.float32); STD = np.array([0.18, 0.18, 0.18], np.float32)


def _x(iid, root):
    a = (np.asarray(Image.open(os.path.join(root, iid)).convert("RGB").resize((W, H)), np.float32) / 255 - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)).astype(np.float32))


@torch.no_grad()
def feats(model, x, dev):
    """(B,3,H,W) → (B, Np, C) L2정규화 패치 임베딩(layer2)."""
    f = model(x.to(dev))[0]                      # (B,C,h,w)
    B, C, h, w = f.shape
    f = f.permute(0, 2, 3, 1).reshape(B, h * w, C)
    return torch.nn.functional.normalize(f, dim=2), (h, w)


def is_normal(iid, ann):
    a = ann.get(iid, {})
    return not any(str(a.get(c, "")).strip() for c in range(1, 5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--normals", type=int, default=1200, help="뱅크 구성용 정상 이미지 수")
    ap.add_argument("--val", type=int, default=1500, help="평가 val 이미지 수")
    ap.add_argument("--per_img", type=int, default=256, help="이미지당 샘플 패치 수")
    ap.add_argument("--bank", type=int, default=30000, help="코어셋 뱅크 크기")
    ap.add_argument("--topk", type=int, default=50, help="이미지 점수=상위 k 패치거리 평균")
    ap.add_argument("--bs", type=int, default=8)
    args = ap.parse_args()
    torch.manual_seed(config.SEED); np.random.seed(config.SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ann = data.load_annotations()
    tr, va = data.split_fold(args.fold)
    normals = [i for i in tr if is_normal(i, ann)][:args.normals]
    va = va[:args.val]
    y = np.array([0 if is_normal(i, ann) else 1 for i in va])
    print(f"[data] 정상뱅크 {len(normals)} / val {len(va)} (결함 {int(y.sum())}, 정상 {int((1-y).sum())})", flush=True)

    model = timm.create_model("resnet18", pretrained=True, features_only=True, out_indices=(2,)).to(dev).eval()

    # ── 뱅크 구성 ──
    t0 = time.time(); pool = []
    for i in range(0, len(normals), args.bs):
        xb = torch.stack([_x(j, config.TRAIN_IMG) for j in normals[i:i + args.bs]])
        fb, _ = feats(model, xb, dev)                       # (B,Np,C)
        fb = fb.reshape(-1, fb.shape[-1]).cpu().numpy()
        idx = np.random.choice(len(fb), min(args.per_img * xb.shape[0], len(fb)), replace=False)
        pool.append(fb[idx])
    pool = np.concatenate(pool)
    if len(pool) > args.bank:
        pool = pool[np.random.choice(len(pool), args.bank, replace=False)]
    bank = torch.from_numpy(pool).to(dev)                   # (M,C)
    print(f"[bank] {bank.shape} ({time.time()-t0:.0f}s)", flush=True)

    # ── 점수 ──
    scores = []
    for i in range(0, len(va), args.bs):
        xb = torch.stack([_x(j, config.TRAIN_IMG) for j in va[i:i + args.bs]])
        fb, _ = feats(model, xb, dev)                       # (B,Np,C)
        for b in range(fb.shape[0]):
            d = torch.cdist(fb[b], bank)                    # (Np,M)
            nn = d.min(1).values                            # 패치별 최근접 거리
            s = torch.topk(nn, min(args.topk, nn.numel())).values.mean().item()
            scores.append(s)
    scores = np.array(scores)

    from sklearn.metrics import roc_auc_score, average_precision_score
    auroc = roc_auc_score(y, scores); ap_ = average_precision_score(y, scores)
    # recall@FP — 정상 95% 통과(=FPR5%) 임계에서 결함 recall
    thr = np.quantile(scores[y == 0], 0.95)
    rec = float((scores[y == 1] >= thr).mean())
    print(f"\n=== ReconPatch(무라벨) 이미지 AUROC {auroc:.4f}  AP {ap_:.4f}  recall@FPR5% {rec:.3f}", flush=True)
    print(f"   (다른 팀 Conv-AE 0.7023 대비 {'+' if auroc>0.7023 else ''}{auroc-0.7023:+.4f})", flush=True)
    out = {"auroc": float(auroc), "ap": float(ap_), "recall_at_fpr5": rec,
           "n_normal_bank": len(normals), "n_val": len(va), "bank": int(bank.shape[0]),
           "vs_ae_0p70": float(auroc - 0.7023)}
    op = os.path.join(config.ROOT, "docs", "overnight", "RECONPATCH_RESULTS.json")
    os.makedirs(os.path.dirname(op), exist_ok=True)
    json.dump(out, open(op, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("saved ->", op, flush=True)


if __name__ == "__main__":
    main()
