"""Stage0 — 패치분류 누수 폭로.

다른 팀처럼 256×1600 이미지를 256×256 슬라이딩 패치로 잘라 '결함/정상' 이진분류.
두 분할을 비교:
  (A) patch-split : 전체 패치를 랜덤 80/20 → 같은 원본의 인접(overlap) 패치가 train/test 동시 → 누수
  (B) image-split : 이미지를 fold로 먼저 나눈 뒤 패치화 → 누수 없음
동일 모델(timm resnet18)로 학습해 정확도/AUC 차이 = 누수가 부풀린 점수.

실행:  python -m src.patch_leak --images 3000 --epochs 4
"""
import os, argparse, time, random
import numpy as np
from PIL import Image
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score
import timm

from src import config, data

PS, STRIDE = 256, 128


def patch_positions(w=config.W, ps=PS, stride=STRIDE):
    xs = list(range(0, w - ps + 1, stride))
    if xs[-1] != w - ps:
        xs.append(w - ps)
    return xs


def build_patch_list(image_ids, ann, min_defect_px=50):
    """-> list of (image_id, x0, label). label=1 결함."""
    xs = patch_positions()
    out = []
    for iid in image_ids:
        union = np.zeros((config.H, config.W), np.uint8)
        for c, rle in ann.get(iid, {}).items():
            union |= data.rle_decode(rle)
        for x0 in xs:
            lab = 1 if union[:, x0:x0 + PS].sum() >= min_defect_px else 0
            out.append((iid, x0, lab))
    return out


class PatchDS(Dataset):
    def __init__(self, items, train):
        self.items = items
        self.train = train
        self._cache = {}

    def __len__(self):
        return len(self.items)

    def _img(self, iid):
        if iid not in self._cache:
            if len(self._cache) > 256:
                self._cache.clear()
            self._cache[iid] = np.array(Image.open(
                os.path.join(config.TRAIN_IMG, iid)).convert("RGB"))
        return self._cache[iid]

    def __getitem__(self, i):
        iid, x0, lab = self.items[i]
        p = self._img(iid)[:, x0:x0 + PS].astype(np.float32) / 255.0
        if self.train and random.random() < 0.5:
            p = p[:, ::-1].copy()
        p = (p - 0.344) / 0.18
        return torch.from_numpy(p.transpose(2, 0, 1)), torch.tensor(lab, dtype=torch.float32)


def run_split(name, tr_items, va_items, epochs, bs, device, workers):
    random.seed(0)
    tr = DataLoader(PatchDS(tr_items, True), batch_size=bs, shuffle=True,
                    num_workers=workers, pin_memory=True, drop_last=True,
                    persistent_workers=workers > 0)
    va = DataLoader(PatchDS(va_items, False), batch_size=bs, shuffle=False,
                    num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    model = timm.create_model("resnet18", pretrained=True, num_classes=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    crit = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        model.train(); t0 = time.time()
        for x, y in tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=True):
                loss = crit(model(x).squeeze(1), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        # eval
        model.eval(); ys, ps = [], []
        with torch.no_grad():
            for x, y in va:
                x = x.to(device, non_blocking=True)
                with torch.autocast("cuda", enabled=True):
                    p = torch.sigmoid(model(x).squeeze(1))
                ps.append(p.float().cpu().numpy()); ys.append(y.numpy())
        y = np.concatenate(ys); p = np.concatenate(ps)
        auc = roc_auc_score(y, p); acc = ((p >= .5) == y).mean()
        f1 = f1_score(y, (p >= .5).astype(int), zero_division=0)
        print(f"  [{name}] ep{ep} auc={auc:.4f} acc={acc:.4f} f1={f1:.4f} ({time.time()-t0:.0f}s)", flush=True)
    return dict(auc=float(auc), acc=float(acc), f1=float(f1), n_tr=len(tr_items), n_va=len(va_items))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=3000, help="사용할 원본 이미지 수")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(config.SEED); np.random.seed(config.SEED)

    ann = data.load_annotations()
    rows = data.load_folds()
    # 이미지 subset(결함/정상 균형 유지 위해 fold 순서 셔플)
    random.shuffle(rows)
    rows = rows[:args.images]
    tr_imgs = [i for i, fo in rows if fo != 0]
    va_imgs = [i for i, fo in rows if fo == 0]

    # (B) image-split
    tr_items_B = build_patch_list(tr_imgs, ann)
    va_items_B = build_patch_list(va_imgs, ann)

    # (A) patch-split: 동일 이미지 풀의 패치를 합쳐 랜덤 80/20
    all_items = tr_items_B + va_items_B
    random.shuffle(all_items)
    cut = int(0.8 * len(all_items))
    tr_items_A, va_items_A = all_items[:cut], all_items[cut:]

    pr = lambda it: f"{len(it)} (결함 {sum(l for _,_,l in it)})"
    print(f"[A patch-split] train {pr(tr_items_A)} / val {pr(va_items_A)}")
    print(f"[B image-split] train {pr(tr_items_B)} / val {pr(va_items_B)}")

    print("\n=== (A) patch-split (누수) ===")
    rA = run_split("A", tr_items_A, va_items_A, args.epochs, args.bs, device, args.workers)
    print("\n=== (B) image-split (정상) ===")
    rB = run_split("B", tr_items_B, va_items_B, args.epochs, args.bs, device, args.workers)

    print("\n================ 누수 폭로 ================")
    print(f"  patch-split AUC {rA['auc']:.4f} acc {rA['acc']:.4f} f1 {rA['f1']:.4f}")
    print(f"  image-split AUC {rB['auc']:.4f} acc {rB['acc']:.4f} f1 {rB['f1']:.4f}")
    print(f"  Δ(누수 인플레)  AUC {rA['auc']-rB['auc']:+.4f}  acc {rA['acc']-rB['acc']:+.4f}  f1 {rA['f1']-rB['f1']:+.4f}")
    import json
    os.makedirs(config.EXP, exist_ok=True)
    with open(os.path.join(config.EXP, "stage0_patch_leak.json"), "w", encoding="utf-8") as f:
        json.dump({"patch_split": rA, "image_split": rB}, f, ensure_ascii=False, indent=2)
    print("saved ->", os.path.join(config.EXP, "stage0_patch_leak.json"))


if __name__ == "__main__":
    main()
