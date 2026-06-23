"""대회 지표 — (이미지×클래스) mean Dice. 빈 GT: 미예측=1, 예측=0."""
import numpy as np


def dice_pair(pred, gt):
    """단일 (이미지,클래스) Dice. 둘 다 빈마스크면 1.0."""
    p, g = pred.astype(bool), gt.astype(bool)
    ps, gs = p.sum(), g.sum()
    if ps == 0 and gs == 0:
        return 1.0
    if ps == 0 or gs == 0:
        return 0.0
    return 2.0 * (p & g).sum() / (ps + gs)


def mean_dice(preds, gts):
    """preds/gts: (N, C, H, W) bool. 대회식 (이미지×클래스) 평균 Dice."""
    N, C = preds.shape[:2]
    s = 0.0
    for n in range(N):
        for c in range(C):
            s += dice_pair(preds[n, c], gts[n, c])
    return s / (N * C)


def per_class_dice(preds, gts):
    N, C = preds.shape[:2]
    out = []
    for c in range(C):
        vals = [dice_pair(preds[n, c], gts[n, c]) for n in range(N)]
        out.append(float(np.mean(vals)))
    return out
