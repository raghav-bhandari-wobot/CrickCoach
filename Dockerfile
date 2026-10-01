# GPU image for the Batsman Analyser (RTX 4060 / Ada -> CUDA 12.8 wheels)
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt && pip uninstall -y opencv-python && pip install opencv-python-headless

COPY batsman_analyser ./batsman_analyser
COPY tools ./tools
COPY analyse.py .

# weights/, videos/ and out/ are mounted as volumes (see docker-compose.yml)
ENTRYPOINT ["python", "analyse.py"]
CMD ["--help"]
