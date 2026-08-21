FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/opt/ml/code

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ml/code
COPY . /opt/ml/code/mlx
RUN python -m pip install --upgrade pip \
    && python -m pip install \
        "boto3>=1.34.0" \
        "PyYAML>=6.0.1" \
        "ultralytics @ git+https://github.com/ralampay/ultralytics" \
        "libreyolo[onnx] @ git+https://github.com/ralampay/libreyolo.git@release"

ENTRYPOINT ["python", "-m", "mlx.modes.object_detection.aws.entrypoint"]
