"""전처리 분리도 분석 (busbar 계승) — 학습 없이 "어떤 전처리가 결함을 잘 살리나" 측정.

각 전처리에 대해, 결함 픽셀 vs 정상 픽셀의 **Cohen's d**(분리도)를 클래스별로 계산해 랭킹.
높은 |d| = 결함이 더 도드라짐 = 학습 스윕에 넣을 가치↑.

실행: python -m src.analyze_preproc --per-class 80
"""
import os, argparse, random
import numpy as np
from PIL import Image
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config, data

DOC_IMG = os.path.join(config.ROOT, "docs", "images")
os.makedirs(DOC_IMG, exist_ok=True)


def cohens_d(a, b):
    if len(a) < 5 or len(b) < 5:
        return 0.0
    va, vb = a.var(), b.var()
    s = np.sqrt((va + vb) / 2) + 1e-6
    return abs(a.mean() - b.mean()) / s


def gray_of(fn, img):
    out = fn(img) if fn else img
    return cv2.cvtColor(out, cv2.COLOR_RGB2GRAY).astype(np.float32)


def texture(g):
    """국소 대비(질감) — |픽셀 - 국소평균|. band-pass에 공정한 분리도용."""
    return np.abs(g - cv2.blur(g, (9, 9)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=80, help="클래스당 표본 이미지 수")
    args = ap.parse_args()
    random.seed(0)
    ann = data.load_annotations()

    # 클래스별 해당 결함 보유 이미지 목록
    by_cls = {c: [] for c in range(1, 5)}
    for iid, d in ann.items():
        for c in d:
            by_cls[c].append(iid)

    methods = list(data.PREPROCS.keys())
    res_i = {m: {c: [] for c in range(1, 5)} for m in methods}  # 밝기 분리도
    res_t = {m: {c: [] for c in range(1, 5)} for m in methods}  # 질감 분리도

    for c in range(1, 5):
        ids = random.sample(by_cls[c], min(args.per_class, len(by_cls[c])))
        for iid in ids:
            img = np.array(Image.open(os.path.join(config.TRAIN_IMG, iid)).convert("RGB"))
            mc = data.rle_decode(ann[iid][c]).astype(bool)
            union = np.zeros(mc.shape, bool)
            for cc, rle in ann[iid].items():
                union |= data.rle_decode(rle).astype(bool)
            normal = ~union
            if mc.sum() < 20 or normal.sum() < 20:
                continue
            ni = np.flatnonzero(normal.ravel())
            ni = np.random.choice(ni, min(len(ni), mc.sum() * 3), replace=False)
            mcr = mc.ravel()
            for m in methods:
                g2 = gray_of(data.PREPROCS[m], img)
                t2 = texture(g2)
                gr, tr = g2.ravel(), t2.ravel()
                res_i[m][c].append(cohens_d(gr[mcr], gr[ni]))
                res_t[m][c].append(cohens_d(tr[mcr], tr[ni]))

    def agg(res):
        out = {}
        for m in methods:
            per = [float(np.mean(res[m][c])) if res[m][c] else 0.0 for c in range(1, 5)]
            out[m] = (per, float(np.mean(per)))
        return out
    AI, AT = agg(res_i), agg(res_t)
    # 랭킹 = 밝기·질감 중 큰 값(둘 중 하나라도 잘 분리하면 유망)
    summary = sorted(methods, key=lambda m: max(AI[m][1], AT[m][1]), reverse=True)

    R = ["# 전처리 분리도 분석 (Cohen's d — 밝기 + 질감, 학습 불필요)\n",
         f"> 결함 vs 정상 픽셀 분리도. 클래스당 표본 {args.per_class}장. **밝기**=평균밝기차, "
         "**질감**=국소대비차(band-pass에 공정). 둘 중 큰 값으로 랭킹. *이건 prior일 뿐, 최종은 학습 스윕.*\n",
         "| 전처리 | 밝기 |d| | 질감 |d| | max | C별 질감(C1/C2/C3/C4) |",
         "|---|---|---|---|---|"]
    for m in summary:
        pi, ai = AI[m]; pt, at = AT[m]
        R.append(f"| {m} | {ai:.2f} | {at:.2f} | **{max(ai,at):.2f}** | {[round(x,2) for x in pt]} |")
    base = max(AI["none"][1], AT["none"][1])
    top = [m for m in summary if m != "none" and max(AI[m][1], AT[m][1]) > base][:8]
    R.append(f"\n→ 기준선(none) max|d| = {base:.2f}. **학습 스윕 권장(none 초과 상위)**: {', '.join(top)}\n")
    R.append("⚠️ 이 지표는 prior(사전탐색)일 뿐 — DoG처럼 평균밝기를 없애는 전처리는 밝기|d|가 낮아도 "
             "질감|d|나 실제 학습에선 유효할 수 있음. **최종 선택은 밤샘 학습 스윕의 per-class 검출률로 결정.**\n")

    plt.figure(figsize=(10, 7))
    ys = summary[::-1]
    ai = [AI[m][1] for m in ys]; at = [AT[m][1] for m in ys]
    yp = np.arange(len(ys))
    plt.barh(yp + 0.2, ai, 0.4, label="밝기 |d|", color="#888")
    plt.barh(yp - 0.2, at, 0.4, label="질감 |d|", color="#4363d8")
    plt.yticks(yp, ys, fontsize=7)
    plt.xlabel("|Cohen's d|"); plt.title("Preprocessing separability (intensity vs texture)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(DOC_IMG, "preproc_separability.png"), dpi=110); plt.close()
    R.append("![sep](images/preproc_separability.png)")

    with open(os.path.join(config.ROOT, "docs", "PREPROC_ANALYSIS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(R))
    print("PREPROC ANALYSIS DONE -> docs/PREPROC_ANALYSIS.md")
    for m in summary:
        print(f"  {m:14s} 밝기|d| {AI[m][1]:.2f}  질감|d| {AT[m][1]:.2f}  max {max(AI[m][1],AT[m][1]):.2f}")
    print(f"\n권장 전처리(상위, prior): {top}")


if __name__ == "__main__":
    main()
