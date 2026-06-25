"""데모 샘플 사전계산 — fold0 val에서 클래스별 대표 이미지 추론 → web/static/samples/.
정적 폴백(백엔드 없이도 데모) + 즉시 로드용. CPU로 충분.
"""
import os, sys, json
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config, data
from web import infer

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "samples")
os.makedirs(OUT, exist_ok=True)


def gt4(iid, ann):
    m = np.zeros((4, config.H, config.W), np.uint8)
    for c in range(1, 5):
        if c in ann.get(iid, {}):
            m[c - 1] = data.rle_decode(ann[iid][c])
    return m


def pick(ann, va_ids, n_defect=15, n_normal=5):
    """클래스 골고루 + 멀티클래스 + 정상 = 약 20장."""
    def cls(iid):
        return {c for c in range(1, 5) if c in ann.get(iid, {}) and str(ann[iid][c]).strip()}
    defects = [i for i in va_ids if cls(i)]
    normals = [i for i in va_ids if not cls(i)]
    chosen, seen = [], set()
    # 1) 클래스별 대표 1장씩
    for c in range(1, 5):
        for iid in defects:
            if c in cls(iid) and iid not in seen:
                chosen.append(iid); seen.add(iid); break
    # 2) 멀티클래스 2장
    m = 0
    for iid in defects:
        if m >= 2: break
        if len(cls(iid)) >= 2 and iid not in seen:
            chosen.append(iid); seen.add(iid); m += 1
    # 3) 나머지 결함으로 n_defect 채우기
    for iid in defects:
        if len([x for x in chosen if x in seen]) >= n_defect: break
        if iid not in seen:
            chosen.append(iid); seen.add(iid)
    chosen = chosen[:n_defect]
    # 4) 정상 n_normal
    chosen += normals[:n_normal]
    return chosen


def main():
    ann = data.load_annotations()
    _, va = data.split_fold(0)
    ids = pick(ann, va)
    print("선택:", ids)
    infer.load_models("cpu")
    samples = []
    for iid in ids:
        g = gt4(iid, ann)
        pil = Image.open(os.path.join(config.TRAIN_IMG, iid))
        r = infer.infer(pil, gt4=g, **infer.POSTPROC, use_gate=True)
        if not r["available"]:
            print("모델 없음:", r.get("error")); return
        # PNG 저장(base64 → 파일)
        import base64
        for key, suffix in [("base_png", ".jpg"), ("overlay_png", "_ov.png")]:
            b = base64.b64decode(r[key].split(",", 1)[1])
            open(os.path.join(OUT, iid.replace(".jpg", "") + suffix), "wb").write(b)
        samples.append({"id": iid, "mean_dice": r["mean_dice"],
                        "per_class": r["per_class"], "clf_probs": r["clf_probs"],
                        "base": iid.replace(".jpg", "") + ".jpg",
                        "overlay": iid.replace(".jpg", "") + "_ov.png"})
        print(f"  {iid} dice={r['mean_dice']} clf={r['clf_probs']}")
    json.dump({"samples": samples}, open(os.path.join(OUT, "samples.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("saved ->", os.path.join(OUT, "samples.json"))


if __name__ == "__main__":
    main()
