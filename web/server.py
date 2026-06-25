"""Steel Inspection Console — FastAPI 백엔드.
- 정적 SPA(web/static) 서빙 + 실추론 API. 모델 없으면 정적 샘플로 폴백.
실행: STEEL_DEVICE=cpu uvicorn web.server:app --host 127.0.0.1 --port 8010
"""
import os, io, json
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from web import infer
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")
app = FastAPI(title="Steel Inspection Console")
_ann = {"a": None}


def _gt4(iid):
    from src import config, data
    if _ann["a"] is None:
        _ann["a"] = data.load_annotations()
    m = np.zeros((4, config.H, config.W), np.uint8)
    for c in range(1, 5):
        if c in _ann["a"].get(iid, {}):
            m[c - 1] = data.rle_decode(_ann["a"][iid][c])
    return m, os.path.join(config.TRAIN_IMG, iid)


@app.get("/api/health")
def health():
    st = infer.load_models()
    return {"models_loaded": bool(st["seg"]), "device": st["device"], "error": st["err"],
            "postproc": infer.POSTPROC, "gate_thr": infer.GATE_THR}


@app.get("/api/samples")
def samples():
    p = os.path.join(STATIC, "samples", "samples.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {"samples": []}


@app.get("/api/experiments")
def experiments():
    return json.load(open(os.path.join(STATIC, "experiments.json"), encoding="utf-8"))


@app.post("/api/infer")
async def api_infer(file: UploadFile = File(None), min_prob: float = Form(0.6),
                    max_prob: float = Form(0.7), min_area: int = Form(600),
                    gate: bool = Form(True)):
    if file is None:
        return JSONResponse({"available": False, "error": "no file"}, status_code=400)
    img = Image.open(io.BytesIO(await file.read()))
    r = infer.infer(img, min_prob=min_prob, max_prob=max_prob,
                    min_area=int(min_area), use_gate=gate)
    return r


@app.get("/api/infer_sample")
def infer_sample(id: str, min_prob: float = 0.6, max_prob: float = 0.7,
                 min_area: int = 600, gate: bool = True):
    try:
        g, path = _gt4(id)
        img = Image.open(path)
        return infer.infer(img, min_prob=min_prob, max_prob=max_prob,
                           min_area=int(min_area), use_gate=gate, gt4=g)
    except Exception as e:
        return JSONResponse({"available": False, "error": str(e)}, status_code=400)


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC), name="static")
