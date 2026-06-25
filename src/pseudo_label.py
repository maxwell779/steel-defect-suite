"""② Pseudo-labeling (2라운드) — 누수 차단.

teacher(현 best deeplab/effb3 + 분류 게이트)로 **test_images(미라벨)** 의사라벨 생성
→ fold0 **train에만** 추가(val 격리) → student 재학습 → 다음 라운드 teacher.

실행:
  python -m src.pseudo_label generate --teacher <dir> --out data/pseudo_r1.csv --limit 3000
  python -m src.pseudo_label train    --pseudo data/pseudo_r1.csv --tag _pseudo_r1 --epochs 12
"""
import os, csv, json, time, argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

from src import config, data
from src.metrics import mean_dice, per_class_dice

H, W = config.H, config.W
TEST_IMG = os.path.join(config.DATA, "test_images")
MEAN = (0.344, 0.344, 0.344); STD = (0.18, 0.18, 0.18)
GATE_THR = [0.3, 0.02, 0.5, 0.0]
WIN = dict(arch="deeplabv3plus", enc="efficientnet-b3")
WIN_DIR = "stage2_seg_deeplabv3plus_efficientnet-b3_f0_sw_f_deeplabv3plus_effb3_x3"


def _norm(img):
    a = (np.asarray(img.convert("RGB").resize((W, H)), np.float32) / 255 - MEAN) / STD
    return np.ascontiguousarray(a.transpose(2, 0, 1))


class _ImgDS(Dataset):
    def __init__(self, paths): self.p = paths
    def __len__(self): return len(self.p)
    def __getitem__(self, i): return _norm(Image.open(self.p[i])), os.path.basename(self.p[i])


# ── 라운드 의사라벨 생성 ──
@torch.no_grad()
def generate(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seg = smp.DeepLabV3Plus(encoder_name=WIN["enc"], encoder_weights=None, in_channels=3, classes=4)
    seg.load_state_dict(torch.load(os.path.join(config.EXP, args.teacher, "best.pt"), map_location="cpu"))
    seg.to(dev).eval()
    import timm
    clf = timm.create_model("efficientnet_b3", pretrained=False, num_classes=4)
    clf.load_state_dict(torch.load(os.path.join(config.EXP, args.clf, "best.pt"), map_location="cpu"))
    clf.to(dev).eval()
    paths = [os.path.join(TEST_IMG, f) for f in sorted(os.listdir(TEST_IMG)) if f.endswith(".jpg")]
    if args.limit: paths = paths[:args.limit]
    ld = DataLoader(_ImgDS(paths), batch_size=args.bs, num_workers=args.workers)
    rows, kept = [], 0
    for x, names in ld:
        x = x.to(dev)
        with torch.autocast("cuda", enabled=dev == "cuda"):
            sp = torch.sigmoid(seg(x)).float().cpu().numpy()
            cp = torch.sigmoid(clf(x)).float().cpu().numpy()
        for n, nm in enumerate(names):
            any_c = False
            for c in range(4):
                pc = sp[n, c]
                if cp[n, c] < GATE_THR[c] or pc.max() < args.max_prob:
                    continue
                m = pc >= args.min_prob
                if m.sum() < args.min_area:
                    continue
                rows.append({"ImageId": nm, "ClassId": c + 1, "EncodedPixels": data.rle_encode(m)})
                any_c = True
            kept += any_c
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ImageId", "ClassId", "EncodedPixels"]); w.writeheader()
        w.writerows(rows)
    print(f"[generate] test {len(paths)}장 → 결함예측 {kept}장, RLE행 {len(rows)} -> {args.out}", flush=True)


# ── 통합 데이터셋(실 train + 의사 test) ──
class ListSeg(Dataset):
    def __init__(self, items, tfm): self.items, self.tfm = items, tfm  # [(path, {c:rle})]
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        path, ann = self.items[i]
        img = np.array(Image.open(path).convert("RGB"))
        mask = np.zeros((H, W, 4), np.uint8)
        for c in range(1, 5):
            if c in ann and str(ann[c]).strip():
                mask[:, :, c - 1] = data.rle_decode(ann[c])
        out = self.tfm(image=img, mask=mask)
        im, mk = out["image"], out["mask"]
        if not torch.is_tensor(im): im = torch.from_numpy(im.transpose(2, 0, 1))
        mk = mk.permute(2, 0, 1).float() if torch.is_tensor(mk) else torch.from_numpy(mk.transpose(2, 0, 1)).float()
        return im, mk, os.path.basename(path)


def _load_pseudo(path):
    d = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d.setdefault(r["ImageId"], {})[int(r["ClassId"])] = r["EncodedPixels"]
    return d


def train(args):
    torch.manual_seed(config.SEED); np.random.seed(config.SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ann = data.load_annotations()
    tr_ids, va_ids = data.split_fold(args.fold)
    # 실 train
    items = [(os.path.join(config.TRAIN_IMG, i), ann.get(i, {})) for i in tr_ids]
    # 의사 test (train에만 추가)
    ps = _load_pseudo(args.pseudo)
    items += [(os.path.join(TEST_IMG, k), v) for k, v in ps.items()]
    print(f"[data] 실train {len(tr_ids)} + 의사 {len(ps)} = {len(items)} / val {len(va_ids)}", flush=True)
    tr = DataLoader(ListSeg(items, data.build_tfms(True)), batch_size=args.bs, shuffle=True,
                    num_workers=args.workers, pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    va = DataLoader(data.SteelSegDataset(va_ids, ann, data.build_tfms(False)), batch_size=args.bs,
                    num_workers=args.workers, pin_memory=True)
    model = smp.DeepLabV3Plus(encoder_name=WIN["enc"], encoder_weights="imagenet", in_channels=3, classes=4).to(dev)
    if args.init:  # 이전 라운드 가중치로 워밍 스타트
        model.load_state_dict(torch.load(os.path.join(config.EXP, args.init, "best.pt"), map_location="cpu"))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")
    pw = torch.tensor([4., 8., 1., 2.], device=dev).view(-1, 1, 1)
    bce = nn.BCEWithLogitsLoss(pos_weight=pw)
    lov = smp.losses.LovaszLoss(mode="multilabel")
    crit = lambda lo, t: 0.5 * bce(lo, t) + 0.5 * lov(lo, t)

    from src.train_seg import evaluate
    name = f"stage2_seg_deeplabv3plus_efficientnet-b3_f{args.fold}{args.tag}"
    outdir = os.path.join(config.EXP, name); os.makedirs(outdir, exist_ok=True)
    best = -1
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tot = 0
        for img, mask, _ in tr:
            img, mask = img.to(dev, non_blocking=True), mask.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=dev == "cuda"):
                loss = crit(model(img), mask)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); tot += loss.item()
        sch.step()
        md, pc, _, _ = evaluate(model, va, dev)
        print(f"  ep{ep:02d} loss={tot/len(tr):.4f} val_dice={md:.4f} pc={[round(x,3) for x in pc]} ({time.time()-t0:.0f}s)", flush=True)
        if md > best:
            best = md; torch.save(model.state_dict(), os.path.join(outdir, "best.pt"))
    model.load_state_dict(torch.load(os.path.join(outdir, "best.pt")))
    base, bpc, _, _ = evaluate(model, va, dev, min_size=0)
    bms, bmd = 0, base
    for ms in [400, 600, 800, 1200]:
        md, _, _, _ = evaluate(model, va, dev, min_size=ms)
        if md > bmd: bmd, bms = md, ms
    json.dump({"name": name, "best_val_dice": base, "best_min_size": bms,
               "best_val_dice_postproc": bmd, "per_class": bpc, "n_pseudo": len(ps)},
              open(os.path.join(outdir, "metrics.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[PSEUDO {args.tag}] val_dice={base:.4f} → min_size={bms} {bmd:.4f}  (vs 단일 0.9514)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--teacher", default=WIN_DIR); g.add_argument("--clf", default="stage1_clf_efficientnet_b3_f0")
    g.add_argument("--out", required=True); g.add_argument("--limit", type=int, default=0)
    g.add_argument("--bs", type=int, default=16); g.add_argument("--workers", type=int, default=5)
    g.add_argument("--min_prob", type=float, default=0.6); g.add_argument("--max_prob", type=float, default=0.7)
    g.add_argument("--min_area", type=int, default=600)
    t = sub.add_parser("train")
    t.add_argument("--pseudo", required=True); t.add_argument("--fold", type=int, default=0)
    t.add_argument("--tag", default="_pseudo_r1"); t.add_argument("--init", default="")
    t.add_argument("--epochs", type=int, default=12); t.add_argument("--bs", type=int, default=16)
    t.add_argument("--lr", type=float, default=3e-4); t.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    (generate if a.cmd == "generate" else train)(a)


if __name__ == "__main__":
    main()
