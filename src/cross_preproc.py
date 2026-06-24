"""전처리 교차검증 — UNet/se 스캔 상위 전처리를 '승리 모델'(DeepLabV3+/effb3)에 적용해
모델 넘어 전이되는지 확인. control = DeepLabV3+/effb3 + none (이미 학습됨, 0.9514).

실행: python -m src.cross_preproc --topk 3 --arch deeplabv3plus --encoder efficientnet-b3
"""
import os, json, glob, subprocess, sys, time
from src import config

PY = sys.executable


def score(m):
    return m.get("best_val_dice_postproc", 0) + 0.5 * m.get("min_recall", 0)


def load(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return None


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--arch", default="deeplabv3plus")
    ap.add_argument("--encoder", default="efficientnet-b3")
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()

    # UNet/se 전처리 스캔 결과 수집
    cands = []
    for d in glob.glob(os.path.join(config.EXP, "*_sw_p_unet_se_resnext50_prep_*")):
        mp = os.path.join(d, "metrics.json")
        m = load(mp)
        if m:
            prep = os.path.basename(d).split("_prep_")[-1]
            cands.append((prep, score(m), m.get("best_val_dice_postproc"), m.get("defect_recall")))
    cands.sort(key=lambda x: x[1], reverse=True)
    print(f"=== UNet/se 전처리 스캔 결과 ({len(cands)}개) ===")
    for prep, sc, dice, rec in cands:
        print(f"  {prep:14s} score {sc:.4f} dice {dice:.4f} recall {rec}")
    top = [c[0] for c in cands[:args.topk]]
    print(f"\n교차검증 대상(top {args.topk}): {top}")
    print(f"control = {args.arch}/{args.encoder} + none (기존 0.9514)\n")

    env = dict(os.environ, PYTHONIOENCODING="utf-8", GPU_GUARD_NEED="18000")
    results = []
    procs = []
    for prep in top:
        name = f"stage2_seg_{args.arch}_{args.encoder}_f{args.fold}_cross_{prep}"
        mp = os.path.join(config.EXP, name, "metrics.json")
        if os.path.exists(mp):
            results.append((prep, load(mp))); print(f"  [skip] {prep}"); continue
        tag = f"_cross_{prep}"
        cmd = [PY, "-m", "src.train_seg", "--fold", str(args.fold), "--arch", args.arch,
               "--encoder", args.encoder, "--loss", "focaltversky", "--preproc", prep,
               "--epochs", str(args.epochs), "--bs", "16", "--workers", "5",
               "--oversample", "--posweight", "4,8,1,2", "--tag", tag]
        log = open(os.path.join(config.ROOT, "logs", name + ".log"), "w")
        print(f"  [run] {name}", flush=True)
        # 2-way 병렬
        procs.append((prep, subprocess.Popen(cmd, env=env, cwd=config.ROOT, stdout=log, stderr=subprocess.STDOUT), mp))
        while sum(1 for _, p, _ in procs if p.poll() is None) >= 2:
            time.sleep(10)
    for prep, p, mp in procs:
        p.wait(); m = load(mp)
        if m:
            results.append((prep, m))

    # 종합
    R = ["# 전처리 교차검증 (UNet/se 상위 전처리 → DeepLabV3+/effb3)\n",
         f"> control = {args.arch}/{args.encoder} + none = **0.9514**. 이걸 넘으면 전처리가 전이됨.\n",
         "| 전처리 | val Dice | 검출률 | vs none(0.9514) |", "|---|---|---|---|"]
    for prep, m in sorted(results, key=lambda x: x[1].get("best_val_dice_postproc", 0), reverse=True):
        d = m.get("best_val_dice_postproc", 0)
        R.append(f"| {prep} | {d:.4f} | {m.get('defect_recall')} | {d-0.9514:+.4f} |")
    R.append("\n→ none(0.9514)을 넘는 전처리가 있으면 최종 모델에 적용, 없으면 **none이 최선**(전처리 무용 확인).")
    out = os.path.join(config.ROOT, "docs", "overnight", "CROSS_PREPROC.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(R))
    print("\nCROSS DONE ->", out)
    for prep, m in results:
        print(f"  {prep}: {m.get('best_val_dice_postproc'):.4f}")


if __name__ == "__main__":
    main()
