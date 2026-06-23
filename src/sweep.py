"""밤샘 스윕 — 프록시(짧은 학습) 스캔 → per-class 검출률 랭킹 → 상위 K 풀학습.

- 큐레이션 그리드(arch×enc + 전처리 + 손실 + 하이퍼파라미터). dict 기반 config(+extra args).
- **--parallel K**: K개 동시 실행(각 GPU_GUARD_NEED 낮춰 메모리 분할). 1=순차.
- 랭킹 = best_val_dice_postproc + 0.5*min_recall(빈마스크 착시 방지).
- subprocess 격리 + metrics.json 있으면 skip(재개) + 실패해도 계속.

실행: python -m src.sweep --proxy-epochs 5 --full-epochs 16 --topk 3 --parallel 2
"""
import os, json, subprocess, argparse, time, sys
from src import config

PY = sys.executable
COMMON = ["--oversample", "--posweight", "4,8,1,2"]  # 공통(하이퍼파라미터 그룹은 override)


def cfg(arch, enc, loss="focaltversky", prep="none", extra=None, label=None):
    return {"arch": arch, "enc": enc, "loss": loss, "prep": prep,
            "extra": extra or [], "label": label or ""}


def build_grid():
    g = []
    se = "se_resnext50_32x4d"
    # ── A. 아키텍처 × se_resnext50 ──
    for a in ["unet", "fpn", "unetpp", "deeplabv3plus", "manet", "pspnet", "linknet", "pan"]:
        g.append(cfg(a, se))
    # ── B. unet × 백본 (대형/이질 포함) ──
    for e in ["efficientnet-b3", "efficientnet-b4", "efficientnet-b5",
              "resnet34", "resnet50", "timm-resnest50d", "tu-regnety_040", "mit_b1"]:
        g.append(cfg("unet", e))
    g.append(cfg("fpn", "efficientnet-b4"))
    g.append(cfg("fpn", "efficientnet-b3"))
    # ── C. 전처리 (unet/se, 분리도 분석 상위 + band-pass) ──
    for p in ["gamma", "pctnorm", "canny", "dog_xwide", "dog_wide", "log", "gabor", "clahe"]:
        g.append(cfg("unet", se, prep=p, label=f"prep_{p}"))
    # ── D. 손실 ──
    for l in ["lovasz", "bce_lovasz", "bcedice"]:
        g.append(cfg("unet", se, loss=l, label=f"loss_{l}"))
    # ── E. 하이퍼파라미터 (unet/se/focaltversky) ──
    g.append(cfg("unet", se, extra=["--lr", "1e-4"], label="lr1e4"))
    g.append(cfg("unet", se, extra=["--lr", "5e-4"], label="lr5e4"))
    g.append(cfg("unet", se, extra=["--posweight", "6,12,1,2"], label="pw6_12"))
    g.append(cfg("unet", se, extra=["--tv-beta", "0.85"], label="tvb085"))
    g.append(cfg("unet", se, extra=["--tv-beta", "0.6"], label="tvb06"))
    # ── F. 크로스 ──
    g.append(cfg("fpn", se, loss="lovasz", label="x1"))
    g.append(cfg("fpn", se, prep="dog_xwide", label="x2"))
    g.append(cfg("deeplabv3plus", "efficientnet-b3", loss="bce_lovasz", label="x3"))
    g.append(cfg("manet", "efficientnet-b3", label="x4"))
    return g


GRID = build_grid()


def name_of(c, phase, fold=0):
    e = c["enc"].replace("_32x4d", "").replace("efficientnet-", "eff").replace("timm-", "").replace("tu-", "")
    lab = ("_" + c["label"]) if c["label"] else f"_{c['loss']}_{c['prep']}"
    return f"stage2_seg_{c['arch']}_{c['enc']}_f{fold}_sw_{phase}_{c['arch']}_{e}{lab}"


def launch(c, epochs, phase, fold, need):
    name = name_of(c, phase, fold)
    mpath = os.path.join(config.EXP, name, "metrics.json")
    if os.path.exists(mpath):
        print(f"  [skip] {name}", flush=True)
        return None, mpath
    env = dict(os.environ, PYTHONIOENCODING="utf-8", GPU_GUARD_NEED=str(need))
    log = open(os.path.join(config.ROOT, "logs", name + ".log"), "w")
    # --tag: name에서 _sw_ 이후를 사용
    tag = "_sw" + name.split("_sw", 1)[1]
    base = [PY, "-m", "src.train_seg", "--fold", str(fold), "--arch", c["arch"],
            "--encoder", c["enc"], "--loss", c["loss"], "--preproc", c["prep"],
            "--epochs", str(epochs), "--bs", "16", "--workers", "5", "--oversample",
            "--tag", tag]
    has_pw = "--posweight" in c["extra"]
    if not has_pw:
        base += ["--posweight", "4,8,1,2"]
    base += c["extra"]
    p = subprocess.Popen(base, env=env, cwd=config.ROOT, stdout=log, stderr=subprocess.STDOUT)
    print(f"  [run] {name}", flush=True)
    return p, mpath


def run_pool(cfgs, epochs, phase, fold, parallel, need):
    """K개 동시 실행 풀."""
    results = []
    running = []  # (proc, mpath, cfg, t0)
    i = 0
    while i < len(cfgs) or running:
        while len(running) < parallel and i < len(cfgs):
            p, mp = launch(cfgs[i], epochs, phase, fold, need)
            if p is None:  # skip(이미완료)
                m = load(mp)
                if m: m["_cfg"] = cfgs[i]; results.append(m)
            else:
                running.append((p, mp, cfgs[i], time.time()))
            i += 1
        # 완료 체크
        time.sleep(5)
        still = []
        for p, mp, c, t0 in running:
            if p.poll() is None:
                still.append((p, mp, c, t0))
            else:
                m = load(mp)
                dt = time.time() - t0
                if m:
                    m["_cfg"] = c; results.append(m)
                    print(f"      ✓ {os.path.basename(os.path.dirname(mp))} "
                          f"dice {m['best_val_dice_postproc']:.4f} recall {m.get('defect_recall')} ({dt:.0f}s)", flush=True)
                else:
                    print(f"      ✗ {os.path.basename(os.path.dirname(mp))} 실패 ({dt:.0f}s)", flush=True)
        running = still
    return results


def load(mp):
    try:
        return json.load(open(mp, encoding="utf-8"))
    except Exception:
        return None


def score(m):
    return m["best_val_dice_postproc"] + 0.5 * m.get("min_recall", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy-epochs", type=int, default=5)
    ap.add_argument("--full-epochs", type=int, default=16)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--need", type=int, default=18000, help="병렬 시 run당 GPU 예약(MiB)")
    args = ap.parse_args()
    os.makedirs(os.path.join(config.ROOT, "logs"), exist_ok=True)
    outdoc = os.path.join(config.ROOT, "docs", "overnight"); os.makedirs(outdoc, exist_ok=True)
    need = args.need if args.parallel > 1 else 30000

    print(f"=== PHASE 1: 프록시 스캔 ({len(GRID)} config, parallel={args.parallel}) ===", flush=True)
    t0 = time.time()
    results = run_pool(GRID, args.proxy_epochs, "p", args.fold, args.parallel, need)
    print(f"[Phase1 완료] {len(results)}/{len(GRID)} ({time.time()-t0:.0f}s)", flush=True)
    results.sort(key=score, reverse=True)
    with open(os.path.join(outdoc, "sweep_proxy.json"), "w", encoding="utf-8") as f:
        json.dump([{"name": m["name"], "dice": m["best_val_dice_postproc"],
                    "recall": m.get("defect_recall"), "score": score(m)} for m in results],
                  f, ensure_ascii=False, indent=2)

    print(f"\n=== PHASE 2: 상위 {args.topk} 풀학습 ===", flush=True)
    full = run_pool([m["_cfg"] for m in results[:args.topk]], args.full_epochs, "f",
                    args.fold, min(args.parallel, args.topk), need)
    full.sort(key=score, reverse=True)

    R = ["# Steel 스윕 결과\n",
         f"> 프록시 {args.proxy_epochs}ep × {len(GRID)}config (parallel={args.parallel}) → 상위 {args.topk} 풀학습. "
         "랭킹=val Dice(후처리)+0.5*최약클래스 검출률.\n", "## 프록시 (score 내림차순)",
         "| config | val Dice | 검출률 C1/C2/C3/C4 | score |", "|---|---|---|---|"]
    for m in results:
        R.append(f"| {m['name'].replace('stage2_seg_','').split('_sw_')[0]} | "
                 f"{m['best_val_dice_postproc']:.4f} | {m.get('defect_recall')} | {score(m):.4f} |")
    R.append("\n## 풀학습 상위")
    R.append("| config | val Dice | 검출률 |"); R.append("|---|---|---|")
    for m in full:
        R.append(f"| {m['name'].replace('stage2_seg_','').split('_sw_')[0]} | "
                 f"{m['best_val_dice_postproc']:.4f} | {m.get('defect_recall')} |")
    if full:
        b = full[0]
        R.append(f"\n**최고**: {b['name']} — Dice {b['best_val_dice_postproc']:.4f}, 검출률 {b.get('defect_recall')}")
    with open(os.path.join(outdoc, "SWEEP_RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(R))
    print("\nSWEEP DONE -> docs/overnight/SWEEP_RESULTS.md", flush=True)


if __name__ == "__main__":
    main()
