# Steel Inspection Console — CPU 추론 컨테이너
FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# torch CPU 휠 + 추론 의존성
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir \
        segmentation-models-pytorch timm fastapi "uvicorn[standard]" \
        python-multipart pillow opencv-python-headless numpy scikit-learn

COPY src/ ./src/
COPY web/ ./web/

ENV STEEL_DEVICE=cpu PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
EXPOSE 8010
# 가중치(experiments/)는 런타임에 볼륨 마운트. 없으면 정적 샘플로 동작.
CMD ["uvicorn", "web.server:app", "--host", "0.0.0.0", "--port", "8010"]
