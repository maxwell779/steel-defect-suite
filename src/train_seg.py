"""Stage2 세그멘테이션 베이스라인 — smp UNet(seresnext50) + BCE+Dice + AMP.
이미지 단위 fold로 누수 없이 학습/검증, 대회식 mean-Dice 산출.

실행:  python -m src.train_seg --fold 0 --encoder se_resnext50_32x4d --epochs 12 --bs 12
"""
import os, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from src import config, data
from src.metrics import mean_dice, per_class_dice


class BCEDice(nn.Module):
    def __init__(self, w_bce=0.6, w_dice=0.4, pos_weight=None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.w_bce, self.w_dice = w_bce, w_dice

    def forward(self, logits, target):
        bce = self.bce(logits, target)
        p = torch.sigmoid(logits)
        dims = (0, 2, 3)
        inter = (p * target).sum(dims)
        denom = p.sum(dims) + target.sum(dims)
        dice = 1 - ((2 * inter + 1) / (denom + 1)).mean()
        return self.w_bce * bce + self.w_dice * dice


class FocalTverskyBCE(nn.Module):
    """희귀·작은 결함용 — Tversky(beta>alpha=FN 더 처벌) + focal + BCE(per-class pos_weight).
    C1/C2 붕괴(전부 빈칸 예측) 방지."""
    def __init__(self, alpha=0.3, beta=0.7, gamma=1.3, w_bce=0.5, w_tv=0.5, pos_weight=None):  # noqa
        super().__init__()
        self.a, self.b, self.g = alpha, beta, gamma
        self.w_bce, self.w_tv = w_bce, w_tv
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, target):
        p = torch.sigmoid(logits)
        dims = (0, 2, 3)
        tp = (p * target).sum(dims)
        fp = (p * (1 - target)).sum(dims)
        fn = ((1 - p) * target).sum(dims)
        ti = (tp + 1) / (tp + self.a * fp + self.b * fn + 1)   # per-class Tversky
        tv = ((1 - ti) ** self.g).mean()
        return self.w_bce * self.bce(logits, target) + self.w_tv * tv


@torch.no_grad()
def evaluate(model, loader, device, thr=0.5, min_size=0):
    model.eval()
    preds_all, gts_all = [], []
    for img, mask, _ in loader:
        img = img.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=True):
            p = torch.sigmoid(model(img))
        p = (p.float().cpu().numpy() >= thr)
        if min_size > 0:
            for n in range(p.shape[0]):
                for c in range(p.shape[1]):
                    if p[n, c].sum() < min_size:
                        p[n, c] = False
        preds_all.append(p)
        gts_all.append(mask.numpy().astype(bool))
    preds = np.concatenate(preds_all)
    gts = np.concatenate(gts_all)
    return mean_dice(preds, gts), per_class_dice(preds, gts), preds, gts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--encoder", default="se_resnext50_32x4d")
    ap.add_argument("--arch", default="unet",
                    choices=["unet", "fpn", "unetpp", "deeplabv3plus", "pspnet", "manet", "linknet", "pan"])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="스모크용 train 표본 상한(0=전체)")
    ap.add_argument("--loss", default="bcedice", choices=["bcedice", "focaltversky", "lovasz", "bce_lovasz"])
    ap.add_argument("--preproc", default="none", choices=list(data.PREPROCS.keys()))
    ap.add_argument("--oversample", action="store_true", help="희귀클래스(C1/C2) 포함 이미지 가중 샘플링")
    ap.add_argument("--posweight", default="", help="클래스별 BCE pos_weight, 예: 4,8,1,2")
    ap.add_argument("--tv-beta", type=float, default=0.7, help="Tversky beta(클수록 FN 더 처벌)")
    ap.add_argument("--tv-gamma", type=float, default=1.3, help="FocalTversky gamma")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    torch.manual_seed(config.SEED); np.random.seed(config.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ann = data.load_annotations()
    tr_ids, va_ids = data.split_fold(args.fold)
    if args.limit:
        tr_ids = tr_ids[:args.limit]; va_ids = va_ids[:max(200, args.limit // 4)]
    print(f"[data] train {len(tr_ids)} / val {len(va_ids)} (fold {args.fold})")

    tr_ds = data.SteelSegDataset(tr_ids, ann, data.build_tfms(True, args.preproc))
    va_ds = data.SteelSegDataset(va_ids, ann, data.build_tfms(False, args.preproc))
    # 희귀클래스 가중 샘플러 — 이미지가 가진 클래스 중 가장 희귀한 것 기준 가중
    if args.oversample:
        from torch.utils.data import WeightedRandomSampler
        freq = {1: 897, 2: 247, 3: 5150, 4: 801}  # 인스턴스 빈도(EDA)
        w = []
        for iid in tr_ids:
            cs = ann.get(iid, {}).keys()
            w.append(max([5150 / freq[c] for c in cs], default=1.0))  # 희귀할수록 가중↑
        sampler = WeightedRandomSampler(w, num_samples=len(tr_ids), replacement=True)
        tr_ld = DataLoader(tr_ds, batch_size=args.bs, sampler=sampler, num_workers=args.workers,
                           pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
        print(f"[sampler] 희귀클래스 오버샘플 ON (가중 max {max(w):.1f})")
    else:
        tr_ld = DataLoader(tr_ds, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                           pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    va_ld = DataLoader(va_ds, batch_size=args.bs, shuffle=False, num_workers=args.workers,
                       pin_memory=True, persistent_workers=args.workers > 0)

    Arch = {"unet": smp.Unet, "fpn": smp.FPN, "unetpp": smp.UnetPlusPlus,
            "deeplabv3plus": smp.DeepLabV3Plus, "pspnet": smp.PSPNet,
            "manet": smp.MAnet, "linknet": smp.Linknet, "pan": smp.PAN}[args.arch]
    model = Arch(encoder_name=args.encoder, encoder_weights="imagenet",
                 in_channels=3, classes=config.N_CLASSES).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")
    pw = None
    if args.posweight:
        pw = torch.tensor([float(x) for x in args.posweight.split(",")], device=device).view(-1, 1, 1)
    if args.loss == "focaltversky":
        crit = FocalTverskyBCE(alpha=1 - args.tv_beta, beta=args.tv_beta, gamma=args.tv_gamma, pos_weight=pw)
    elif args.loss in ("lovasz", "bce_lovasz"):
        from segmentation_models_pytorch.losses import LovaszLoss
        lov = LovaszLoss(mode="multilabel")
        if args.loss == "lovasz":
            crit = lov
        else:
            bce = nn.BCEWithLogitsLoss(pos_weight=pw)
            crit = lambda lo, t: 0.5 * bce(lo, t) + 0.5 * lov(lo, t)
    else:
        crit = BCEDice(pos_weight=pw)
    print(f"[loss] {args.loss}  preproc={args.preproc}  pos_weight={args.posweight or 'none'}")

    name = f"stage2_seg_{args.arch}_{args.encoder}_f{args.fold}{args.tag}"
    outdir = os.path.join(config.EXP, name); os.makedirs(outdir, exist_ok=True)
    best = -1; hist = []
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tot = 0
        for img, mask, _ in tr_ld:
            img, mask = img.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=True):
                loss = crit(model(img), mask)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item()
        sched.step()
        md, pc, _, _ = evaluate(model, va_ld, device)
        hist.append({"epoch": ep, "loss": tot / len(tr_ld), "val_dice": md,
                     "per_class": pc})
        print(f"  ep{ep:02d} loss={tot/len(tr_ld):.4f} val_dice={md:.4f} "
              f"per_class={[round(x,3) for x in pc]} ({time.time()-t0:.0f}s)", flush=True)
        if md > best:
            best = md
            torch.save(model.state_dict(), os.path.join(outdir, "best.pt"))

    # min-size 후처리 튜닝(val)
    model.load_state_dict(torch.load(os.path.join(outdir, "best.pt")))
    base, base_pc, _, _ = evaluate(model, va_ld, device, thr=0.5, min_size=0)
    best_ms, best_ms_dice = 0, base
    for ms in [200, 400, 600, 800, 1200]:
        md, _, _, _ = evaluate(model, va_ld, device, thr=0.5, min_size=ms)
        if md > best_ms_dice:
            best_ms_dice, best_ms = md, ms
    print(f"\n[BEST] val_dice={base:.4f} (no postproc) → min_size={best_ms} "
          f"val_dice={best_ms_dice:.4f}")
    # per-class 검출률(decompose) — 빈마스크 착시 방지용 진짜 지표
    from src.analyze_seg import decompose
    _, _, P, G = evaluate(model, va_ld, device, thr=0.5, min_size=best_ms)
    dec = decompose(P, G)
    recalls = [round(d["defect_recall"], 4) for d in dec]
    print(f"[per-class 검출률] {recalls}  (C1,C2가 0이면 실패)")
    result = {"name": name, "fold": args.fold, "encoder": args.encoder, "arch": args.arch,
              "loss": args.loss, "preproc": args.preproc,
              "best_val_dice": base, "best_per_class": base_pc,
              "best_min_size": best_ms, "best_val_dice_postproc": best_ms_dice,
              "defect_recall": recalls,
              "min_recall": float(min(recalls)),  # 가장 약한 클래스(랭킹용)
              "history": hist}
    with open(os.path.join(outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("SEG DONE ->", outdir)


if __name__ == "__main__":
    main()
