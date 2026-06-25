"""후처리(3-임계) + TTA + 앙상블 — 학습 없이 추가 이득 (상위 솔루션 표준 트릭).

- 앙상블: 풀학습 상위 모델들의 sigmoid 확률 평균
- TTA: 원본/h-flip/v-flip 평균
- 3-임계 후처리(per-class, val 튜닝):
    max_prob_thresh = 마스크 내 최대확률 < t 면 통째로 버림(빈 이미지 게이트)
    min_prob_thresh = 픽셀 이진화 임계
    min_area_thresh = 작은 마스크 제거
- 비교: 단일 best / 앙상블 / 앙상블+TTA / 앙상블+TTA+후처리

실행: python -m src.postprocess --tta
"""
import os, json, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from src import config, data
from src.metrics import dice_pair

# (arch, encoder, 실험 디렉터리명) — 풀학습 상위 모델
MEMBERS = [
    ("deeplabv3plus", "efficientnet-b3", "stage2_seg_deeplabv3plus_efficientnet-b3_f0_sw_f_deeplabv3plus_effb3_x3"),
    ("deeplabv3plus", "efficientnet-b3", "stage2_seg_deeplabv3plus_efficientnet-b3_f0_pseudo_r2"),  # ② pseudo R2(0.9540)
    ("fpn", "se_resnext50_32x4d", "stage2_seg_fpn_se_resnext50_32x4d_f0_sw_f_fpn_se_resnext50_focaltversky_none"),
    ("unet", "se_resnext50_32x4d", "stage2_seg_unet_se_resnext50_32x4d_f0_m2_balanced"),
    ("unet", "se_resnext50_32x4d", "stage2_seg_unet_se_resnext50_32x4d_f0_sw_f_unet_se_resnext50_tvb085"),
]
ARCH = {"unet": smp.Unet, "fpn": smp.FPN, "unetpp": smp.UnetPlusPlus,
        "deeplabv3plus": smp.DeepLabV3Plus, "manet": smp.MAnet, "pspnet": smp.PSPNet}


@torch.no_grad()
def predict_probs(model, loader, device, tta):
    """sigmoid 확률 (float16, CPU). TTA= 원본/hflip/vflip 평균."""
    outs = []
    for img, _, _ in loader:
        img = img.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=True):
            p = torch.sigmoid(model(img))
            if tta:
                p = p + torch.flip(torch.sigmoid(model(torch.flip(img, [3]))), [3])  # hflip
                p = p + torch.flip(torch.sigmoid(model(torch.flip(img, [2]))), [2])  # vflip
                p = p / 3
        outs.append(p.half().cpu())
    return torch.cat(outs).numpy()  # (N,C,H,W) float16


def gts_of(loader):
    g = []
    for _, mask, _ in loader:
        g.append(mask.bool().numpy())
    return np.concatenate(g)  # (N,C,H,W) bool


def mean_dice_from_pred(pred_bool, gts):
    N, C = pred_bool.shape[:2]
    s = 0.0
    for n in range(N):
        for c in range(C):
            s += dice_pair(pred_bool[n, c], gts[n, c])
    return s / (N * C)


def per_class_dice(pred_bool, gts):
    N, C = pred_bool.shape[:2]
    return [float(np.mean([dice_pair(pred_bool[n, c], gts[n, c]) for n in range(N)])) for c in range(C)]


def apply_postproc(probs, min_prob, max_prob, min_area):
    """3-임계 후처리 → bool mask. probs (N,C,H,W) float."""
    N, C = probs.shape[:2]
    out = np.zeros(probs.shape, bool)
    for n in range(N):
        for c in range(C):
            pc = probs[n, c]
            if pc.max() < max_prob:        # 게이트: 충분히 확신하는 픽셀 없으면 빈마스크
                continue
            m = pc >= min_prob
            if m.sum() < min_area:         # 작은 마스크 제거
                continue
            out[n, c] = m
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="val 표본 상한(메모리/속도)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ann = data.load_annotations()
    _, va_ids = data.split_fold(args.fold)
    if args.limit:
        va_ids = va_ids[:args.limit]
    va = DataLoader(data.SteelSegDataset(va_ids, ann, data.build_tfms(False)),
                    batch_size=args.bs, shuffle=False, num_workers=args.workers)
    print(f"[data] val {len(va_ids)}")
    gts = gts_of(va)

    # 멤버별 확률 + 앙상블(누적 평균, float32 누적)
    ens = None; singles = []
    members = [(a, e, d) for a, e, d in MEMBERS if os.path.exists(os.path.join(config.EXP, d, "best.pt"))]
    for a, e, d in members:
        m = ARCH[a](encoder_name=e, encoder_weights=None, in_channels=3, classes=4).to(device).eval()
        m.load_state_dict(torch.load(os.path.join(config.EXP, d, "best.pt"), map_location=device))
        probs = predict_probs(m, va, device, args.tta).astype(np.float32)
        sd = mean_dice_from_pred(probs >= 0.5, gts)
        singles.append((d, sd))
        print(f"  단일 {d.split('_f0')[0].replace('stage2_seg_','')}: dice {sd:.4f} (thr0.5{' +TTA' if args.tta else ''})")
        ens = probs if ens is None else ens + probs
        del m; torch.cuda.empty_cache()
    ens /= len(members)

    # 앙상블 기본(thr0.5)
    ens_base = mean_dice_from_pred(ens >= 0.5, gts)
    print(f"\n앙상블({len(members)}){' +TTA' if args.tta else ''} thr0.5: {ens_base:.4f}")

    # 3-임계 후처리 그리드(val 튜닝)
    best = (ens_base, (0.5, 0.0, 0));
    for min_prob in [0.4, 0.5, 0.6]:
        for max_prob in [0.5, 0.6, 0.7]:
            for min_area in [0, 600, 1200, 2000]:
                pred = apply_postproc(ens, min_prob, max_prob, min_area)
                md = mean_dice_from_pred(pred, gts)
                if md > best[0]:
                    best = (md, (min_prob, max_prob, min_area))
    md_best, (mp, xp, ma) = best
    pred_best = apply_postproc(ens, mp, xp, ma)
    pc = per_class_dice(pred_best, gts)

    print(f"\n=== 최종 (앙상블{'+TTA' if args.tta else ''}+후처리) ===")
    print(f"  best mean Dice {md_best:.4f}  (min_prob={mp}, max_prob={xp}, min_area={ma})")
    print(f"  per-class C1/C2/C3/C4: {[round(x,3) for x in pc]}")

    result = {"members": [d for _, _, d in members], "tta": args.tta,
              "single_best": max(s for _, s in singles),
              "ensemble_thr05": ens_base, "ensemble_postproc": md_best,
              "postproc": {"min_prob": mp, "max_prob": xp, "min_area": ma},
              "per_class": pc, "singles": singles}
    out = os.path.join(config.ROOT, "docs", "overnight", "POSTPROC_RESULTS.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("saved ->", out)


if __name__ == "__main__":
    main()
