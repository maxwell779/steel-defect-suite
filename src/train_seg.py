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
    def __init__(self, w_bce=0.6, w_dice=0.4):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.w_bce, self.w_dice = w_bce, w_dice

    def forward(self, logits, target):
        bce = self.bce(logits, target)
        p = torch.sigmoid(logits)
        dims = (0, 2, 3)
        inter = (p * target).sum(dims)
        denom = p.sum(dims) + target.sum(dims)
        dice = 1 - ((2 * inter + 1) / (denom + 1)).mean()
        return self.w_bce * bce + self.w_dice * dice


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
    ap.add_argument("--arch", default="unet", choices=["unet", "fpn", "unetpp"])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="스모크용 train 표본 상한(0=전체)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    torch.manual_seed(config.SEED); np.random.seed(config.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ann = data.load_annotations()
    tr_ids, va_ids = data.split_fold(args.fold)
    if args.limit:
        tr_ids = tr_ids[:args.limit]; va_ids = va_ids[:max(200, args.limit // 4)]
    print(f"[data] train {len(tr_ids)} / val {len(va_ids)} (fold {args.fold})")

    tr_ds = data.SteelSegDataset(tr_ids, ann, data.build_tfms(True))
    va_ds = data.SteelSegDataset(va_ids, ann, data.build_tfms(False))
    tr_ld = DataLoader(tr_ds, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                       pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    va_ld = DataLoader(va_ds, batch_size=args.bs, shuffle=False, num_workers=args.workers,
                       pin_memory=True, persistent_workers=args.workers > 0)

    Arch = {"unet": smp.Unet, "fpn": smp.FPN, "unetpp": smp.UnetPlusPlus}[args.arch]
    model = Arch(encoder_name=args.encoder, encoder_weights="imagenet",
                 in_channels=3, classes=config.N_CLASSES).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")
    crit = BCEDice()

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
    result = {"name": name, "fold": args.fold, "encoder": args.encoder, "arch": args.arch,
              "best_val_dice": base, "best_per_class": base_pc,
              "best_min_size": best_ms, "best_val_dice_postproc": best_ms_dice,
              "history": hist}
    with open(os.path.join(outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("SEG DONE ->", outdir)


if __name__ == "__main__":
    main()
