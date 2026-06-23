"""밤샘 스윕 — 프록시(짧은 학습) 스캔 → per-class 검출률 기준 랭킹 → 상위 K 풀학습.

설계:
- 세그는 비싸므로(1run~45분) 랜덤 수백개 대신 **큐레이션 그리드**.
- 순차 실행(세그는 GPU 연산 포화 → 병렬 무의미).
- 랭킹 = mean Dice 단독 금지 → **best_val_dice_postproc + 0.5*min_recall**(빈마스크 착시 방지, C1/C2 살린 것 우대).
- subprocess 격리 + metrics.json 있으면 skip(재개 가능) + 한 config 실패해도 계속.

실행: python -m src.sweep --proxy-epochs 6 --full-epochs 16 --topk 3
"""
import os, json, subprocess, argparse, time, sys

from src import config

PY = sys.executable
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", GPU_GUARD_NEED="30000")

# (arch, encoder, loss, preproc)  — oversample/posweight 공통 적용
# 3축 큐레이션: ①아키텍처×인코더 ②전처리(분리도 분석 prior 반영) ③손실
GRID = [
    # ── ① 아키텍처 × 인코더 (focaltversky / none) ──
    ("unet",          "se_resnext50_32x4d", "focaltversky", "none"),  # M2 기준
    ("fpn",           "se_resnext50_32x4d", "focaltversky", "none"),  # FPN 단일 최강 후보
    ("unetpp",        "se_resnext50_32x4d", "focaltversky", "none"),
    ("deeplabv3plus", "se_resnext50_32x4d", "focaltversky", "none"),
    ("manet",         "se_resnext50_32x4d", "focaltversky", "none"),
    ("pspnet",        "se_resnext50_32x4d", "focaltversky", "none"),
    ("unet",          "efficientnet-b3",    "focaltversky", "none"),
    ("fpn",           "efficientnet-b3",    "focaltversky", "none"),
    ("unet",          "efficientnet-b4",    "focaltversky", "none"),
    ("unet",          "resnet34",           "focaltversky", "none"),
    ("unet",          "timm-resnest50d",    "focaltversky", "none"),
    # ── ② 전처리 스윕 (unet/se_resnext50, 분리도 분석 상위 + band-pass) ──
    ("unet",          "se_resnext50_32x4d", "focaltversky", "gamma"),
    ("unet",          "se_resnext50_32x4d", "focaltversky", "pctnorm"),
    ("unet",          "se_resnext50_32x4d", "focaltversky", "canny"),
    ("unet",          "se_resnext50_32x4d", "focaltversky", "dog_xwide"),
    ("unet",          "se_resnext50_32x4d", "focaltversky", "dog_wide"),
    ("unet",          "se_resnext50_32x4d", "focaltversky", "log"),
    ("unet",          "se_resnext50_32x4d", "focaltversky", "gabor"),
    ("unet",          "se_resnext50_32x4d", "focaltversky", "clahe"),
    # ── ③ 손실 비교 (unet/se_resnext50 / none) ──
    ("unet",          "se_resnext50_32x4d", "lovasz",       "none"),
    ("unet",          "se_resnext50_32x4d", "bce_lovasz",   "none"),
    ("unet",          "se_resnext50_32x4d", "bcedice",      "none"),
    # ── 교차 (유망 조합) ──
    ("fpn",           "se_resnext50_32x4d", "lovasz",       "none"),
    ("fpn",           "se_resnext50_32x4d", "focaltversky", "dog_xwide"),
    ("deeplabv3plus", "efficientnet-b3",    "bce_lovasz",   "none"),
]


def tag(arch, enc, loss, prep, phase):
    e = enc.replace("_32x4d", "").replace("efficientnet-", "eff").replace("-", "")
    return f"_sw_{phase}_{arch}_{e}_{loss}_{prep}"


def run(arch, enc, loss, prep, epochs, phase, fold=0, bs=16):
    t = tag(arch, enc, loss, prep, phase)
    name = f"stage2_seg_{arch}_{enc}_f{fold}{t}"
    mpath = os.path.join(config.EXP, name, "metrics.json")
    if os.path.exists(mpath):
        print(f"  [skip] {name} (이미 완료)", flush=True)
        return mpath
    cmd = [PY, "-m", "src.train_seg", "--fold", str(fold), "--arch", arch,
           "--encoder", enc, "--loss", loss, "--preproc", prep,
           "--epochs", str(epochs), "--bs", str(bs), "--workers", "6",
           "--oversample", "--posweight", "4,8,1,2", "--tag", t]
    print(f"  [run] {name} (epochs={epochs})", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, env=ENV, cwd=config.ROOT,
                       stdout=open(os.path.join(config.ROOT, "logs", name + ".log"), "w"),
                       stderr=subprocess.STDOUT)
    print(f"      exit={r.returncode} ({time.time()-t0:.0f}s)", flush=True)
    return mpath if os.path.exists(mpath) else None


def load(mpath):
    try:
        return json.load(open(mpath, encoding="utf-8"))
    except Exception:
        return None


def score(m):
    # 빈마스크 착시 방지: mean Dice + 0.5*최약클래스 검출률
    return m["best_val_dice_postproc"] + 0.5 * m.get("min_recall", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy-epochs", type=int, default=6)
    ap.add_argument("--full-epochs", type=int, default=16)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(os.path.join(config.ROOT, "logs"), exist_ok=True)
    outdoc = os.path.join(config.ROOT, "docs", "overnight"); os.makedirs(outdoc, exist_ok=True)

    print("=== PHASE 1: 프록시 스캔 ===", flush=True)
    results = []
    for cfg in GRID:
        mp = run(*cfg, args.proxy_epochs, "p", args.fold)
        m = load(mp) if mp else None
        if m:
            m["_cfg"] = cfg; results.append(m)
            print(f"      → dice {m['best_val_dice_postproc']:.4f} recall {m['defect_recall']} score {score(m):.4f}", flush=True)

    results.sort(key=score, reverse=True)
    with open(os.path.join(outdoc, "sweep_proxy.json"), "w", encoding="utf-8") as f:
        json.dump([{**{k: m[k] for k in ("name", "best_val_dice_postproc", "defect_recall", "min_recall")},
                    "score": score(m)} for m in results], f, ensure_ascii=False, indent=2)

    print(f"\n=== PHASE 2: 상위 {args.topk} 풀학습 ===", flush=True)
    full = []
    for m in results[:args.topk]:
        cfg = m["_cfg"]
        mp = run(*cfg, args.full_epochs, "f", args.fold)
        fm = load(mp) if mp else None
        if fm:
            fm["_cfg"] = cfg; full.append(fm)
            print(f"      → FULL dice {fm['best_val_dice_postproc']:.4f} recall {fm['defect_recall']}", flush=True)
    full.sort(key=score, reverse=True)

    # 결과 종합 MD
    R = ["# Steel 스윕 결과 (프록시→풀학습)\n",
         f"> 프록시 {args.proxy_epochs}ep {len(results)}config → 상위 {args.topk} 풀학습 {args.full_epochs}ep. "
         "랭킹=val Dice(후처리)+0.5*최약클래스 검출률.\n",
         "## 프록시 스캔 (score 내림차순)",
         "| config | val Dice | 검출률 C1/C2/C3/C4 | score |", "|---|---|---|---|"]
    for m in results:
        R.append(f"| {m['name'].split('_f0')[0].replace('stage2_seg_','')} {m['_cfg'][2]}/{m['_cfg'][3]} "
                 f"| {m['best_val_dice_postproc']:.4f} | {m['defect_recall']} | {score(m):.4f} |")
    R.append("\n## 풀학습 (상위 K)")
    R.append("| config | val Dice | 검출률 C1/C2/C3/C4 |"); R.append("|---|---|---|")
    for m in full:
        R.append(f"| {m['name'].split('_f0')[0].replace('stage2_seg_','')} {m['_cfg'][2]}/{m['_cfg'][3]} "
                 f"| {m['best_val_dice_postproc']:.4f} | {m['defect_recall']} |")
    if full:
        b = full[0]
        R.append(f"\n**최고**: {b['name']} — val Dice {b['best_val_dice_postproc']:.4f}, 검출률 {b['defect_recall']}")
    with open(os.path.join(outdoc, "SWEEP_RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(R))
    print("\nSWEEP DONE -> docs/overnight/SWEEP_RESULTS.md", flush=True)


if __name__ == "__main__":
    main()
