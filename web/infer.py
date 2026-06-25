"""공용 추론 — 세그(승자 DeepLabV3+/effb3) + 분류 게이트 → 4색 마스크 + per-class.

server.py / precompute_samples.py 공용. 모델 없으면 None 반환(정적 폴백).
"""
import os, io, base64
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "experiments")
H, W = 256, 1600
MEAN = np.array([0.344, 0.344, 0.344], np.float32)
STD = np.array([0.18, 0.18, 0.18], np.float32)
GATE_THR = [0.3, 0.02, 0.5, 0.0]                 # fold0 튜닝(고정)
POSTPROC = dict(min_prob=0.6, max_prob=0.7, min_area=600)
COLORS = [(239, 68, 68), (34, 197, 94), (59, 130, 246), (234, 179, 8)]  # C1~C4 R/G/B/Y
CLASS_NAMES = ["C1", "C2", "C3", "C4"]
SEG_DIR = "stage2_seg_deeplabv3plus_efficientnet-b3_f0_sw_f_deeplabv3plus_effb3_x3"
CLF_NAME = "stage1_clf_efficientnet_b3_f0"

_state = {"seg": None, "clf": None, "device": None, "loaded": False, "err": None}


def load_models(device=None):
    if _state["loaded"]:
        return _state
    try:
        import torch, timm
        import segmentation_models_pytorch as smp
        dev = device or os.environ.get("STEEL_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        seg = smp.DeepLabV3Plus(encoder_name="efficientnet-b3", encoder_weights=None,
                                in_channels=3, classes=4)
        seg.load_state_dict(torch.load(os.path.join(EXP, SEG_DIR, "best.pt"), map_location="cpu"))
        seg.to(dev).eval()
        clf = timm.create_model("efficientnet_b3", pretrained=False, num_classes=4)
        clf.load_state_dict(torch.load(os.path.join(EXP, CLF_NAME, "best.pt"), map_location="cpu"))
        clf.to(dev).eval()
        _state.update(seg=seg, clf=clf, device=dev, loaded=True)
    except Exception as e:  # 폴백
        _state.update(loaded=True, err=str(e))
    return _state


def _prep(pil_img):
    from PIL import Image
    img = pil_img.convert("RGB").resize((W, H))
    a = (np.asarray(img, np.float32) / 255.0 - MEAN) / STD
    return np.ascontiguousarray(a.transpose(2, 0, 1))[None], np.asarray(img)


def infer(pil_img, min_prob=0.6, max_prob=0.7, min_area=600, use_gate=True, gt4=None):
    """반환: dict(per_class, base_png, overlay_png, available). 모델없으면 available=False."""
    import torch
    st = load_models()
    if not st["seg"]:
        return {"available": False, "error": st["err"]}
    x, base_rgb = _prep(pil_img)
    dev = st["device"]
    with torch.no_grad():
        xt = torch.from_numpy(x).to(dev)
        seg_p = torch.sigmoid(st["seg"](xt))[0].cpu().numpy()          # (4,H,W)
        clf_p = torch.sigmoid(st["clf"](xt))[0].cpu().numpy()          # (4,)
    overlay = np.zeros((H, W, 4), np.uint8)
    per_class, class_overlays, gt_overlays = [], [], []
    for c in range(4):
        pc = seg_p[c]
        gated = use_gate and clf_p[c] < GATE_THR[c]
        if gated or pc.max() < max_prob:
            mask = np.zeros((H, W), bool)
        else:
            mask = pc >= min_prob
            if mask.sum() < min_area:
                mask = np.zeros((H, W), bool)
        overlay[mask] = (*COLORS[c], 150)
        layer = np.zeros((H, W, 4), np.uint8); layer[mask] = (*COLORS[c], 180)
        class_overlays.append(_png(layer))
        info = {"cls": CLASS_NAMES[c], "present_prob": round(float(clf_p[c]), 4),
                "gated_off": bool(gated), "area": int(mask.sum()),
                "max_prob": round(float(pc.max()), 4), "color": COLORS[c]}
        if gt4 is not None:
            g = gt4[c].astype(bool)
            inter = (mask & g).sum()
            info["dice"] = round(float((2 * inter + 1) / (mask.sum() + g.sum() + 1)), 4)
            info["gt_present"] = bool(g.any())
            gl = np.zeros((H, W, 4), np.uint8); gl[g] = (*COLORS[c], 180)
            gt_overlays.append(_png(gl))
        per_class.append(info)
    return {"available": True, "per_class": per_class,
            "clf_probs": [round(float(p), 4) for p in clf_p],
            "base_png": _png(base_rgb), "overlay_png": _png(overlay),
            "class_overlays": class_overlays,
            "gt_overlays": gt_overlays if gt4 is not None else None,
            "mean_dice": round(float(np.mean([p["dice"] for p in per_class])), 4)
            if gt4 is not None else None}


def _png(arr):
    from PIL import Image
    mode = "RGBA" if arr.ndim == 3 and arr.shape[2] == 4 else "RGB"
    buf = io.BytesIO()
    Image.fromarray(arr, mode).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
