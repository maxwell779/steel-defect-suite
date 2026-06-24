"""분류 게이트 (M5) — 멀티라벨 4-logit '이미지에 클래스c 결함이 있나'.

세그 앞단에서 empty 이미지/클래스를 차단해 대회 Dice의 빈마스크 FP를 억제한다.
- 이미지 단위 fold(누수 차단, data.split_fold 재사용), per-class AUROC/F1.
- 게이트 효과: val에서 '이 클래스 없음' 판정 시 세그 예측을 끄면 empty-FP가 얼마나 주는지
  추정하기 위해, 클래스별 '결함 recall 95% 유지하는 임계'와 그때 'empty 정확 차단율'을 산출.

실행: python -m src.train_clf --fold 0 --encoder efficientnet_b3 --epochs 12 --bs 16
"""
import os, json, time, argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import timm

from src import config, data


def _present(rle):
    return bool(rle) and rle == rle and str(rle).strip() != ""


class SteelClfDataset(Dataset):
    """이미지(3ch) + 멀티라벨 4-logit(클래스별 결함 유무)."""
    def __init__(self, image_ids, ann, tfm):
        self.ids, self.ann, self.tfm = image_ids, ann, tfm

    def __len__(self):
        return len(self.ids)

    def label(self, iid):
        a = self.ann.get(iid, {})
        return np.array([1.0 if _present(a.get(c)) else 0.0
                         for c in range(1, config.N_CLASSES + 1)], np.float32)

    def __getitem__(self, idx):
        iid = self.ids[idx]
        img = np.array(Image.open(os.path.join(config.TRAIN_IMG, iid)).convert("RGB"))
        out = self.tfm(image=img)
        x = out["image"]
        if not torch.is_tensor(x):
            x = torch.from_numpy(x.transpose(2, 0, 1))
        return x, torch.from_numpy(self.label(iid)), iid


def clf_tfms(train=True):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    mean, std = (0.344, 0.344, 0.344), (0.18, 0.18, 0.18)
    aug = [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
           A.RandomBrightnessContrast(p=0.5)] if train else []
    return A.Compose(aug + [A.Normalize(mean, std), ToTensorV2()])


@torch.no_grad()
def evaluate(model, loader, device):
    from sklearn.metrics import roc_auc_score, f1_score
    model.eval()
    P, Y = [], []
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=True):
            p = torch.sigmoid(model(x))
        P.append(p.float().cpu().numpy()); Y.append(y.numpy())
    P, Y = np.concatenate(P), np.concatenate(Y)
    aurocs, f1s, gate = [], [], []
    for c in range(config.N_CLASSES):
        yc, pc = Y[:, c], P[:, c]
        au = roc_auc_score(yc, pc) if yc.min() != yc.max() else float("nan")
        f1 = f1_score(yc, (pc >= 0.5).astype(int)) if yc.max() > 0 else float("nan")
        # 결함 recall 95% 유지하는 임계 → 그때 empty(정상) 정확 차단율
        pos = np.sort(pc[yc == 1])
        thr = pos[max(0, int(0.05 * len(pos)) - 1)] if len(pos) else 0.5
        neg = pc[yc == 0]
        suppress = float((neg < thr).mean()) if len(neg) else 0.0
        aurocs.append(round(float(au), 4)); f1s.append(round(float(f1), 4))
        gate.append({"thr": round(float(thr), 4), "empty_block@recall95": round(suppress, 4)})
    return aurocs, f1s, gate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--encoder", default="efficientnet_b3")  # timm 이름
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="스모크용 train 상한(0=전체)")
    ap.add_argument("--posweight", default="4,8,1,2")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    torch.manual_seed(config.SEED); np.random.seed(config.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ann = data.load_annotations()
    tr_ids, va_ids = data.split_fold(args.fold)
    if args.limit:
        tr_ids = tr_ids[:args.limit]; va_ids = va_ids[:max(200, args.limit // 4)]
    print(f"[data] train {len(tr_ids)} / val {len(va_ids)} (fold {args.fold})", flush=True)

    tr_ds = SteelClfDataset(tr_ids, ann, clf_tfms(True))
    va_ds = SteelClfDataset(va_ids, ann, clf_tfms(False))
    # 희귀클래스 가중 샘플러(seg와 동일 정신)
    freq = {1: 897, 2: 247, 3: 5150, 4: 801}
    w = []
    for iid in tr_ids:
        cs = [c for c in range(1, 5) if _present(ann.get(iid, {}).get(c))]
        w.append(max([5150 / freq[c] for c in cs], default=1.0))
    sampler = WeightedRandomSampler(w, num_samples=len(tr_ids), replacement=True)
    tr_ld = DataLoader(tr_ds, batch_size=args.bs, sampler=sampler, num_workers=args.workers,
                       pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    va_ld = DataLoader(va_ds, batch_size=args.bs, shuffle=False, num_workers=args.workers,
                       pin_memory=True, persistent_workers=args.workers > 0)

    model = timm.create_model(args.encoder, pretrained=True, num_classes=config.N_CLASSES).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")
    pw = torch.tensor([float(x) for x in args.posweight.split(",")], device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    print(f"[clf] {args.encoder} 멀티라벨4 pos_weight={args.posweight}", flush=True)

    name = f"stage1_clf_{args.encoder}_f{args.fold}{args.tag}"
    outdir = os.path.join(config.EXP, name); os.makedirs(outdir, exist_ok=True)
    best, hist = -1, []
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tot = 0
        for x, y, _ in tr_ld:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=True):
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item()
        sched.step()
        au, f1, gate = evaluate(model, va_ld, device)
        mau = float(np.nanmean(au))
        hist.append({"epoch": ep, "loss": tot / len(tr_ld), "auroc": au, "mean_auroc": mau})
        print(f"  ep{ep:02d} loss={tot/len(tr_ld):.4f} mAUROC={mau:.4f} "
              f"auroc={au} f1={f1} ({time.time()-t0:.0f}s)", flush=True)
        if mau > best:
            best = mau
            torch.save(model.state_dict(), os.path.join(outdir, "best.pt"))
            best_state = {"auroc": au, "f1": f1, "gate": gate, "mean_auroc": mau}

    result = {"name": name, "fold": args.fold, "encoder": args.encoder,
              "mean_auroc": best, **best_state, "history": hist}
    with open(os.path.join(outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[BEST] mAUROC={best:.4f} per-class={best_state['auroc']}", flush=True)
    print(f"[게이트] {best_state['gate']}", flush=True)
    print("CLF DONE ->", outdir, flush=True)


if __name__ == "__main__":
    main()
