"""대시보드 집계 사전계산 → web/static/dashboard.json (실측값).
class 분포·val 검사 KPI·per-class 검출률 등. 백엔드 없이도(정적) 동작하게 정적 파일로.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config, data

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "dashboard.json")
NAMES = {1: "Class 1", 2: "Class 2", 3: "Class 3", 4: "Class 4"}
COLORS = {1: "#ef4444", 2: "#22c55e", 3: "#3b82f6", 4: "#f59e0b"}


def _present(r):
    return bool(r) and r == r and str(r).strip() != ""


def main():
    ann = data.load_annotations()
    tr, va = data.split_fold(0)
    # 클래스별 이미지 수(전체 train)
    img_cnt = {c: 0 for c in range(1, 5)}
    multi = 0
    for iid, a in ann.items():
        cs = [c for c in range(1, 5) if _present(a.get(c))]
        for c in cs:
            img_cnt[c] += 1
        if len(cs) >= 2:
            multi += 1
    n_total_imgs = 12568
    class_dist = [{"cls": f"C{c}", "name": NAMES[c], "count": img_cnt[c],
                   "color": COLORS[c]} for c in range(1, 5)]
    # val(fold0) 검사 KPI
    va_defect = sum(1 for i in va if any(_present(ann.get(i, {}).get(c)) for c in range(1, 5)))
    kpis = {
        "total": len(va), "defect": va_defect, "normal": len(va) - va_defect,
        "defect_rate": round(100 * va_defect / len(va), 1),
        "normal_rate": round(100 * (len(va) - va_defect) / len(va), 1),
        "mean_dice_fold0": 0.9573, "oof_dice": 0.9532,
        "train_images": n_total_imgs, "empty_pair_pct": 85.9, "multilabel_imgs": multi,
    }
    # per-class 검출률(실측, RESULTS M2/최종)
    detection = {"labels": ["C1", "C2", "C3", "C4"],
                 "baseline": [0.0, 0.0, 93.4, 96.2],
                 "balanced": [83.4, 93.9, 92.8, 98.1]}
    out = {"kpis": kpis, "class_dist": class_dist, "detection": detection}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("saved", OUT, "| class_imgs", img_cnt, "| val", kpis["total"], "defect", va_defect)


if __name__ == "__main__":
    main()
