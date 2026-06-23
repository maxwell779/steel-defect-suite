"""세그 모델 심층 분석 — 대회 Dice를 분해(빈마스크 억제 / 결함 검출 / 분할 품질),
하드케이스·임계·min-size 민감도·예측 시각화.

실행: python -m src.analyze_seg --fold 0 --encoder se_resnext50_32x4d
"""
import os, json, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config, data
from src.metrics import dice_pair

DOC_IMG = os.path.join(config.ROOT, "docs", "images")
os.makedirs(DOC_IMG, exist_ok=True)
COL = {0: "#e6194B", 1: "#3cb44b", 2: "#4363d8", 3: "#f58231"}


@torch.no_grad()
def infer(model, loader, device, thr=0.5, min_size=0):
    P, G, ids = [], [], []
    for img, mask, iid in loader:
        with torch.autocast("cuda", enabled=True):
            p = torch.sigmoid(model(img.to(device)))
        p = (p.float().cpu().numpy() >= thr)
        if min_size > 0:
            for n in range(p.shape[0]):
                for c in range(p.shape[1]):
                    if p[n, c].sum() < min_size:
                        p[n, c] = False
        P.append(p); G.append(mask.numpy().astype(bool)); ids += list(iid)
    return np.concatenate(P), np.concatenate(G), ids


def decompose(preds, gts):
    """클래스별 대회 Dice 분해."""
    N, C = preds.shape[:2]
    out = []
    for c in range(C):
        tn = fp = fn = det = 0; det_dice = []
        for n in range(N):
            ge = gts[n, c].sum() == 0
            pe = preds[n, c].sum() == 0
            if ge and pe: tn += 1
            elif ge and not pe: fp += 1
            elif not ge and pe: fn += 1
            else:
                det += 1; det_dice.append(dice_pair(preds[n, c], gts[n, c]))
        n_empty = tn + fp; n_def = fn + det
        mean_dice = (tn + sum(det_dice)) / N
        out.append(dict(cls=c + 1, n_empty=n_empty, n_def=n_def,
                        empty_suppress=tn / max(1, n_empty),     # 빈마스크 정확 억제율
                        empty_FP=fp,                              # 빈데 결함 예측(오검출)
                        defect_recall=det / max(1, n_def),        # 결함 검출률
                        defect_FN=fn,                             # 결함 놓침
                        seg_quality=float(np.mean(det_dice)) if det_dice else 0.0,  # 검출된 것의 분할 Dice
                        mean_dice=mean_dice))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--encoder", default="se_resnext50_32x4d")
    ap.add_argument("--arch", default="unet")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ann = data.load_annotations()
    _, va_ids = data.split_fold(args.fold)
    va = DataLoader(data.SteelSegDataset(va_ids, ann, data.build_tfms(False)),
                    batch_size=args.bs, shuffle=False, num_workers=args.workers)
    Arch = {"unet": smp.Unet, "fpn": smp.FPN, "unetpp": smp.UnetPlusPlus}[args.arch]
    model = Arch(encoder_name=args.encoder, encoder_weights=None,
                 in_channels=3, classes=config.N_CLASSES).to(device).eval()
    name = f"stage2_seg_{args.arch}_{args.encoder}_f{args.fold}"
    model.load_state_dict(torch.load(os.path.join(config.EXP, name, "best.pt"), map_location=device))

    preds, gts, ids = infer(model, va, device, thr=0.5, min_size=800)
    dec = decompose(preds, gts)

    R = ["# 세그 모델 심층 분석 (fold0, UNet/se_resnext50)\n",
         "> 대회 Dice를 **빈마스크 억제 / 결함 검출 / 분할 품질**로 분해. thr=0.5, min_size=800.\n",
         "## 1. 클래스별 Dice 분해",
         "| 클래스 | mean Dice | 빈마스크 억제율 | empty FP | 결함 검출률 | 결함 FN | 검출분할 품질 |",
         "|---|---|---|---|---|---|---|"]
    for d in dec:
        R.append(f"| C{d['cls']} | {d['mean_dice']:.3f} | {d['empty_suppress']*100:.1f}% "
                 f"({d['n_empty']}) | {d['empty_FP']} | {d['defect_recall']*100:.1f}% "
                 f"({d['n_def']}) | {d['defect_FN']} | {d['seg_quality']:.3f} |")
    overall = np.mean([d["mean_dice"] for d in dec])
    R.append(f"\n→ 전체 mean Dice **{overall:.4f}**. "
             "**해석**: 빈마스크 억제율(=점수의 86% 차지)이 대부분 높고, "
             "결함 검출률·분할 품질이 클래스별 천장을 가른다.\n")

    # 하드케이스 — 캡처별 평균 dice 최저
    cap_dice = []
    for n in range(len(ids)):
        ds = [dice_pair(preds[n, c], gts[n, c]) for c in range(config.N_CLASSES)]
        cap_dice.append((ids[n], float(np.mean(ds)), ds))
    cap_dice.sort(key=lambda x: x[1])
    R.append("## 2. 최난 이미지 (평균 Dice 최저 10)")
    R.append("| ImageId | 평균 Dice | per-class |")
    R.append("|---|---|---|")
    for iid, md, ds in cap_dice[:10]:
        R.append(f"| {iid} | {md:.3f} | {[round(x,2) for x in ds]} |")
    R.append("")

    # 임계/ min-size 민감도
    R.append("## 3. 임계(thr) · min-size 민감도")
    R.append("| thr | min_size | mean Dice |")
    R.append("|---|---|---|")
    for thr in [0.3, 0.5, 0.7]:
        for ms in [0, 800, 1500]:
            p2, g2, _ = infer(model, va, device, thr=thr, min_size=ms)
            md = np.mean([d2["mean_dice"] for d2 in decompose(p2, g2)])
            R.append(f"| {thr} | {ms} | {md:.4f} |")
    R.append("")

    # 시각화 — 하드케이스 6장 GT(초록) vs Pred(빨강)
    sel = [c[0] for c in cap_dice[:6]]
    from PIL import Image
    fig, axes = plt.subplots(6, 1, figsize=(13, 10))
    id2idx = {iid: i for i, iid in enumerate(ids)}
    for ax, iid in zip(axes, sel):
        n = id2idx[iid]
        im = np.array(Image.open(os.path.join(config.TRAIN_IMG, iid)).convert("RGB"))
        ov = im.copy()
        for c in range(config.N_CLASSES):
            g = gts[n, c]; p = preds[n, c]
            ov[g & ~p] = [0, 255, 0]      # GT만(놓침) = 초록
            ov[p & ~g] = [255, 0, 0]      # Pred만(오검출) = 빨강
            ov[g & p] = [255, 255, 0]     # 일치 = 노랑
        md = float(np.mean([dice_pair(preds[n, c], gts[n, c]) for c in range(config.N_CLASSES)]))
        ax.imshow(ov); ax.set_title(f"{iid}  dice={md:.3f}  (green=FN / red=FP / yellow=hit)", fontsize=8)
        ax.axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(DOC_IMG, "seg_hardcases.png"), dpi=100); plt.close()
    R.append("## 4. 최난 이미지 시각화")
    R.append("![hardcases](images/seg_hardcases.png)")
    R.append("\n초록=GT 놓침(FN) / 빨강=오검출(FP) / 노랑=정확. 남은 오류 양상 확인용.\n")

    with open(os.path.join(config.ROOT, "docs", "ANALYSIS_seg.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(R))
    print("ANALYSIS DONE -> docs/ANALYSIS_seg.md")
    for d in dec:
        print(f"  C{d['cls']}: dice {d['mean_dice']:.3f} | 억제 {d['empty_suppress']*100:.1f}% FP{d['empty_FP']} "
              f"| 검출 {d['defect_recall']*100:.1f}% FN{d['defect_FN']} | 분할 {d['seg_quality']:.3f}")


if __name__ == "__main__":
    main()
