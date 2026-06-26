"""ONNX export + 지연(p50/p95/p99) + int8 동적양자화 — steel seg(DeepLabV3+/effb3)·clf(EffNet-B3).
turbofan MLOps 층 포팅(세그/분류용). 출처: pytorch onnx, onnxruntime quantization.

사용: python -m src.mlops.export_onnx
"""
from __future__ import annotations
import json, time, os
import numpy as np
import torch
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from web.infer import load_models, W, H, EXP

OUT = os.path.join(EXP, "mlops"); os.makedirs(OUT, exist_ok=True)
ONNX = os.path.join(EXP, "onnx"); os.makedirs(ONNX, exist_ok=True)


def _bench(fn, x, warmup=5, iters=40):
    for _ in range(warmup): fn(x)
    ts = []
    for _ in range(iters):
        t = time.perf_counter(); fn(x); ts.append((time.perf_counter() - t) * 1000)
    a = np.array(ts)
    return {"p50": round(float(np.percentile(a, 50)), 2), "p95": round(float(np.percentile(a, 95)), 2),
            "p99": round(float(np.percentile(a, 99)), 2), "mean": round(float(a.mean()), 2)}


def export_one(net, name, dummy):
    fp = os.path.join(ONNX, f"{name}.onnx"); q8 = os.path.join(ONNX, f"{name}_int8.onnx")
    torch.onnx.export(net, dummy, fp, input_names=["x"], output_names=["y"],
                      dynamic_axes={"x": {0: "b"}}, opset_version=17, dynamo=False)
    quantize_dynamic(fp, q8, weight_type=QuantType.QInt8)
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    s = ort.InferenceSession(fp, so, providers=["CPUExecutionProvider"])
    s8 = ort.InferenceSession(q8, so, providers=["CPUExecutionProvider"])
    xnp = dummy.numpy()
    with torch.no_grad():
        y_pt = net(dummy).numpy()
    y_on = s.run(None, {"x": xnp})[0]
    return {
        "size_mb": {"onnx": round(os.path.getsize(fp) / 1e6, 1), "int8": round(os.path.getsize(q8) / 1e6, 1)},
        "parity_max_abs_err": round(float(np.abs(y_pt - y_on).max()), 5),
        "latency_ms_cpu_bs1": {
            "pytorch": _bench(lambda x: net(torch.tensor(x)).detach(), xnp),
            "onnx": _bench(lambda x: s.run(None, {"x": x}), xnp),
            "onnx_int8": _bench(lambda x: s8.run(None, {"x": x}), xnp)},
    }


def run():
    st = load_models(device="cpu")
    if st.get("err") or st.get("seg") is None:
        print("모델 로드 실패:", st.get("err")); return
    seg, clf = st["seg"], st["clf"]
    dummy = torch.randn(1, 3, H, W)
    res = {"seg(DeepLabV3+/effb3)": export_one(seg, "seg", dummy),
           "clf(EffNet-B3 gate)": export_one(clf, "clf", dummy)}
    json.dump(res, open(os.path.join(OUT, "latency.json"), "w"), indent=2)
    for k, v in res.items():
        L = v["latency_ms_cpu_bs1"]
        print(f"[onnx] {k}: parity={v['parity_max_abs_err']} | p50(ms) pt={L['pytorch']['p50']} onnx={L['onnx']['p50']} int8={L['onnx_int8']['p50']} | {v['size_mb']['onnx']}→{v['size_mb']['int8']}MB", flush=True)


if __name__ == "__main__":
    os.environ["STEEL_DEVICE"] = "cpu"; run()
