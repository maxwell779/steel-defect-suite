"""Severstal Steel Defect Detection — 상세 EDA + leak-free fold 생성.

- train.csv(RLE) 파싱 → 클래스/면적/멀티라벨/빈마스크 통계
- 대회 지표(이미지×클래스 mean-Dice) 관점의 빈마스크 비율
- 이미지 단위 StratifiedKFold(클래스조합 기준) → data/folds.csv (누수 방지)
- 그림 저장 → docs/images/

실행: python -m src.eda   (repo 루트에서)
"""
import os, io, csv, json
from collections import defaultdict, Counter
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
IMG_DIR = os.path.join(DATA, "train_images")
DOC_IMG = os.path.join(ROOT, "docs", "images")
os.makedirs(DOC_IMG, exist_ok=True)

H, W = 256, 1600
PIX = H * W
CLASS_NAMES = {1: "C1 (점상/얼룩)", 2: "C2 (희귀)", 3: "C3 (대형/스크래치)", 4: "C4 (압흔)"}
COLORS = {1: "#e6194B", 2: "#3cb44b", 3: "#4363d8", 4: "#f58231"}


def rle_decode(rle, h=H, w=W):
    """Severstal RLE(열 우선, 1-base) → (h,w) mask."""
    s = list(map(int, rle.split()))
    starts, lengths = s[0::2], s[1::2]
    mask = np.zeros(h * w, dtype=np.uint8)
    for st, ln in zip(starts, lengths):
        mask[st - 1: st - 1 + ln] = 1
    return mask.reshape((w, h)).T  # 열 우선이라 (w,h)->T


def rle_area(rle):
    s = list(map(int, rle.split()))
    return sum(s[1::2])


def load_train():
    rows = []
    with open(os.path.join(DATA, "train.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append((row["ImageId"], int(row["ClassId"]), row["EncodedPixels"]))
    return rows


def main():
    rows = load_train()
    all_imgs = sorted(f for f in os.listdir(IMG_DIR) if f.endswith(".jpg"))
    img_classes = defaultdict(set)
    cls_count = Counter()
    cls_area = defaultdict(list)
    for img, cid, rle in rows:
        img_classes[img].add(cid)
        cls_count[cid] += 1
        cls_area[cid].append(rle_area(rle))

    labeled = set(img_classes)
    clean = set(all_imgs) - labeled
    tot_rows = sum(cls_count.values())

    R = []
    R.append("# EDA — Severstal Steel Defect Detection\n")
    R.append(f"> 자동 생성: `python -m src.eda`. 이미지 {H}×{W} 그레이.\n")

    R.append("## 1. 이미지 수")
    R.append(f"- train 총 **{len(all_imgs):,}장**")
    R.append(f"- 결함 있는 이미지: **{len(labeled):,}** ({100*len(labeled)/len(all_imgs):.1f}%)")
    R.append(f"- 정상(결함 0) 이미지: **{len(clean):,}** ({100*len(clean)/len(all_imgs):.1f}%)\n")

    R.append("## 2. 클래스별 결함 인스턴스 (불균형)")
    R.append("| 클래스 | 개수 | 비율 |")
    R.append("|---|---|---|")
    for c in [1, 2, 3, 4]:
        R.append(f"| {CLASS_NAMES[c]} | {cls_count[c]:,} | {100*cls_count[c]/tot_rows:.1f}% |")
    R.append(f"| **합계** | **{tot_rows:,}** | 100% |")
    R.append(f"\n→ **극심한 불균형**: C3가 {100*cls_count[3]/tot_rows:.0f}%, C2는 {100*cls_count[2]/tot_rows:.0f}%뿐. "
             f"불균형비 ≈ **{cls_count[3]/cls_count[2]:.0f}:1**.\n")

    R.append("## 3. 멀티라벨 (한 이미지에 결함 종류 수)")
    ml = Counter(len(cs) for cs in img_classes.values())
    R.append("| 결함 종류 수 | 이미지 | ")
    R.append("|---|---|")
    for k in sorted(ml):
        R.append(f"| {k}종 | {ml[k]:,} |")
    R.append(f"\n→ **{sum(v for k,v in ml.items() if k>=2):,}장이 2종 이상** 동시 보유 → 단일라벨 4-분류는 문제정의 오류. **멀티라벨 마스크 필수**.\n")

    R.append("## 4. 대회 지표(이미지×클래스 mean-Dice) 관점")
    total_pairs = len(all_imgs) * 4
    R.append(f"- 전체 (이미지×클래스) 쌍 = {len(all_imgs):,}×4 = **{total_pairs:,}**")
    R.append(f"- 결함 있는 쌍: **{tot_rows:,}** ({100*tot_rows/total_pairs:.2f}%)")
    R.append(f"- **빈 마스크 쌍: {total_pairs-tot_rows:,} ({100*(total_pairs-tot_rows)/total_pairs:.2f}%)** — 미예측 시 Dice=1, 한 픽셀이라도 예측하면 Dice=0")
    R.append(f"\n→ 점수의 **{100*(total_pairs-tot_rows)/total_pairs:.0f}%가 '빈 마스크를 비워두기'**에서 나옴. **empty FP 억제(분류 게이트)가 핵심**.\n")

    R.append("## 5. 클래스별 마스크 면적(픽셀)")
    R.append("| 클래스 | n | median | mean | min | max | 이미지대비 median |")
    R.append("|---|---|---|---|---|---|---|")
    for c in [1, 2, 3, 4]:
        a = np.array(cls_area[c])
        R.append(f"| {CLASS_NAMES[c]} | {len(a):,} | {np.median(a):,.0f} | {a.mean():,.0f} | "
                 f"{a.min():,} | {a.max():,} | {100*np.median(a)/PIX:.2f}% |")
    R.append("\n→ C3는 넓고, C1/C2는 작은 결함 → 작은 마스크일수록 후처리(min-size)·해상도 민감.\n")

    # ---- 5b. min-size 후처리 가이드 (면적 백분위) ----
    R.append("### 5b. min-size 후처리 가이드 (클래스별 면적 백분위)")
    R.append("| 클래스 | p1 | p5 | p10 | p25 | → min-size 후보 |")
    R.append("|---|---|---|---|---|---|")
    for c in [1, 2, 3, 4]:
        a = np.array(cls_area[c])
        p = np.percentile(a, [1, 5, 10, 25])
        R.append(f"| C{c} | {p[0]:,.0f} | {p[1]:,.0f} | {p[2]:,.0f} | {p[3]:,.0f} | ~p5({p[1]:,.0f}) 이하 제거 검토 |")
    R.append("\n→ 예측 마스크가 클래스별 p5 면적보다 작으면 거짓양성일 확률↑ → val에서 min-size 튜닝.\n")

    # ---- 5c. 연결요소(결함 덩어리 수) ----
    try:
        from scipy import ndimage
        comp_counts = defaultdict(list)
        for img, cid, rle in rows:
            m = rle_decode(rle)
            _, n = ndimage.label(m)
            comp_counts[cid].append(n)
        R.append("### 5c. 마스크당 연결요소(결함 덩어리) 수")
        R.append("| 클래스 | median | mean | max |")
        R.append("|---|---|---|---|")
        for c in [1, 2, 3, 4]:
            cc = np.array(comp_counts[c])
            R.append(f"| C{c} | {np.median(cc):.0f} | {cc.mean():.1f} | {cc.max()} |")
        R.append("\n→ C1/C3는 한 이미지에 덩어리 다수(점상·산발) → 인스턴스 분리/후처리 영향.\n")
    except Exception as e:
        R.append(f"\n(연결요소 분석 생략: {e})\n")

    # ---- 5d. 클래스 동시발생 행렬 ----
    R.append("### 5d. 클래스 동시발생 (co-occurrence)")
    co = np.zeros((4, 4), dtype=int)
    for cs in img_classes.values():
        cl = sorted(cs)
        for i in cl:
            for j in cl:
                co[i-1, j-1] += 1
    R.append("| | C1 | C2 | C3 | C4 |")
    R.append("|---|---|---|---|---|")
    for i in range(4):
        R.append(f"| C{i+1} | " + " | ".join(str(co[i, j]) for j in range(4)) + " |")
    R.append("\n→ 대각=단독, 비대각=동시출현. 멀티라벨 학습 근거.\n")

    # ---- 5e. 공간 분포 히트맵 (마스크 빈도) ----
    try:
        DS = 8  # downsample
        heat = {c: np.zeros((H // DS, W // DS), dtype=np.float64) for c in [1, 2, 3, 4]}
        for img, cid, rle in rows:
            m = rle_decode(rle)[::DS, ::DS]
            heat[cid][:m.shape[0], :m.shape[1]] += m
        fig, axes = plt.subplots(4, 1, figsize=(12, 7))
        for ax, c in zip(axes, [1, 2, 3, 4]):
            ax.imshow(heat[c] / max(1, cls_count[c]), aspect="auto", cmap="hot")
            ax.set_title(f"C{c} mask frequency (n={cls_count[c]})", fontsize=9); ax.axis("off")
        plt.tight_layout(); plt.savefig(os.path.join(DOC_IMG, "eda_spatial.png"), dpi=100); plt.close()
        # 좌우/중앙 편향 정량
        R.append("### 5e. 공간 분포 (결함 위치 편향)")
        R.append("![spatial](images/eda_spatial.png)")
        R.append("\n| 클래스 | 좌1/3 | 중1/3 | 우1/3 |")
        R.append("|---|---|---|---|")
        for c in [1, 2, 3, 4]:
            colsum = heat[c].sum(axis=0)
            w3 = colsum.shape[0] // 3
            l, m_, r = colsum[:w3].sum(), colsum[w3:2*w3].sum(), colsum[2*w3:].sum()
            t = l + m_ + r + 1e-9
            R.append(f"| C{c} | {100*l/t:.0f}% | {100*m_/t:.0f}% | {100*r/t:.0f}% |")
        R.append("\n→ 위치 편향이 크면 ROI/crop 전략, 균일하면 풀이미지 학습 유리.\n")
    except Exception as e:
        R.append(f"\n(공간분포 생략: {e})\n")

    # ---- 5f. 밝기 누수 검증 (다른 팀 주장 재현) ----
    try:
        from PIL import Image
        import random
        random.seed(0)
        def sample_brightness(imglist, k=400):
            vals = []
            for im in random.sample(imglist, min(k, len(imglist))):
                a = np.asarray(Image.open(os.path.join(IMG_DIR, im)).convert("L"))
                vals.append(a.mean())
            return np.array(vals)
        # 클래스별 단독 이미지의 밝기
        single = defaultdict(list)
        for img, cs in img_classes.items():
            if len(cs) == 1:
                single[next(iter(cs))].append(img)
        R.append("### 5f. 밝기 누수 검증 (다른 팀 'RD밝고 LS어둡다' 주장 재현)")
        R.append("| 그룹 | 표본 평균밝기 | std |")
        R.append("|---|---|---|")
        cb = sample_brightness(list(clean))
        R.append(f"| 정상 | {cb.mean():.1f} | {cb.std():.1f} |")
        for c in [1, 2, 3, 4]:
            if single[c]:
                b = sample_brightness(single[c])
                R.append(f"| C{c} 단독 | {b.mean():.1f} | {b.std():.1f} |")
        R.append("\n→ 클래스 간 밝기 평균차가 크면 모델이 **밝기 자체를 학습할 위험**(다른 팀 지적). "
                 "대응: 밝기정규화(CLAHE)·밝기증강으로 결함 텍스처에 집중.\n")
    except Exception as e:
        R.append(f"\n(밝기 분석 생략: {e})\n")

    # ---- folds (image-level, stratified by class-combo) ----
    combo = {}
    for img in all_imgs:
        cs = tuple(sorted(img_classes.get(img, ())))
        combo[img] = cs if cs else (0,)
    # stratify key as string
    keys = {img: "_".join(map(str, combo[img])) for img in all_imgs}
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X = np.array(all_imgs)
    y = np.array([keys[i] for i in all_imgs])
    # rare combos -> merge for stratify safety
    kc = Counter(y)
    y2 = np.array([k if kc[k] >= 5 else "rare" for k in y])
    folds = np.zeros(len(X), dtype=int)
    for fi, (_, va) in enumerate(skf.split(X, y2)):
        folds[va] = fi
    with open(os.path.join(DATA, "folds.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["ImageId", "fold", "classes", "has_defect"])
        for i, img in enumerate(X):
            cs = combo[img]
            w.writerow([img, folds[i], "" if cs == (0,) else "|".join(map(str, cs)),
                        0 if cs == (0,) else 1])
    R.append("## 6. Leak-free 분할 (이미지 단위)")
    R.append("- **이미지 단위 5-fold StratifiedKFold**(클래스조합 기준), `data/folds.csv` 저장.")
    R.append("- 패치는 학습편의일 뿐 — 평가는 원본 이미지 단위 → **인접 패치 누수 원천 차단**.")
    fold_defect = defaultdict(lambda: [0, 0])
    for i, img in enumerate(X):
        fold_defect[folds[i]][0 if combo[img] == (0,) else 1] += 1
    R.append("\n| fold | 정상 | 결함 |")
    R.append("|---|---|---|")
    for fi in range(5):
        R.append(f"| {fi} | {fold_defect[fi][0]:,} | {fold_defect[fi][1]:,} |")
    R.append("")

    # ---- figures ----
    # fig1: class distribution
    plt.figure(figsize=(7, 4))
    cs = [1, 2, 3, 4]
    plt.bar([f"C{c}" for c in cs], [cls_count[c] for c in cs],
            color=[COLORS[c] for c in cs])
    plt.title("Class distribution (instances)"); plt.ylabel("count")
    for i, c in enumerate(cs):
        plt.text(i, cls_count[c], f"{cls_count[c]:,}", ha="center", va="bottom")
    plt.tight_layout(); plt.savefig(os.path.join(DOC_IMG, "eda_class_dist.png"), dpi=110); plt.close()

    # fig2: area distribution (log)
    plt.figure(figsize=(7, 4))
    plt.boxplot([np.log10(np.array(cls_area[c]) + 1) for c in cs],
                labels=[f"C{c}" for c in cs])
    plt.title("Mask area distribution (log10 pixels)"); plt.ylabel("log10(area)")
    plt.tight_layout(); plt.savefig(os.path.join(DOC_IMG, "eda_area_dist.png"), dpi=110); plt.close()

    # fig3: defect vs clean + multilabel
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].pie([len(labeled), len(clean)], labels=["defect", "clean"],
              autopct="%1.0f%%", colors=["#e6194B", "#cccccc"])
    ax[0].set_title("Images: defect vs clean")
    mk = sorted(ml)
    ax[1].bar([str(k) for k in mk], [ml[k] for k in mk], color="#4363d8")
    ax[1].set_title("Defect types per image"); ax[1].set_xlabel("# classes")
    plt.tight_layout(); plt.savefig(os.path.join(DOC_IMG, "eda_overview.png"), dpi=110); plt.close()

    # fig4: sample masks overlay
    try:
        from PIL import Image
        samples = []
        seen = set()
        for img, cid, rle in rows:
            if cid not in seen:
                samples.append((img, cid, rle)); seen.add(cid)
            if len(seen) == 4:
                break
        fig, axes = plt.subplots(4, 1, figsize=(12, 8))
        for ax, (img, cid, rle) in zip(axes, samples):
            im = np.array(Image.open(os.path.join(IMG_DIR, img)).convert("RGB"))
            m = rle_decode(rle)
            overlay = im.copy()
            col = tuple(int(COLORS[cid][i:i+2], 16) for i in (1, 3, 5))
            overlay[m == 1] = (0.5 * np.array(col) + 0.5 * overlay[m == 1]).astype(np.uint8)
            ax.imshow(overlay); ax.set_title(f"{img}  Class {cid}", fontsize=9); ax.axis("off")
        plt.tight_layout(); plt.savefig(os.path.join(DOC_IMG, "eda_samples.png"), dpi=100); plt.close()
        R.append("## 7. 샘플 (클래스별 마스크 오버레이)")
        R.append("![samples](images/eda_samples.png)\n")
    except Exception as e:
        R.append(f"\n(샘플 오버레이 생략: {e})\n")

    R.append("## 그림")
    R.append("![class](images/eda_class_dist.png)")
    R.append("![area](images/eda_area_dist.png)")
    R.append("![overview](images/eda_overview.png)")

    with open(os.path.join(ROOT, "docs", "EDA.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(R))

    summary = dict(total=len(all_imgs), defect=len(labeled), clean=len(clean),
                   cls_count=dict(cls_count), multilabel=dict(ml),
                   empty_pair_ratio=round((total_pairs-tot_rows)/total_pairs, 4))
    with open(os.path.join(ROOT, "docs", "eda_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("EDA DONE ->", os.path.join(ROOT, "docs", "EDA.md"))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
