# B_baseline · 低空水上救援 baseline（x86 训练/评测/导出端）
# 用于 RTX4070S 训练机；需 nvidia-container-toolkit，运行加 --gpus all。
# 注意：Jetson Orin Nano 的 TensorRT 推理(trt_infer_orin.py)在板上 JetPack 原生跑，不用本镜像。
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# OpenCV / ultralytics 运行所需系统库
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*

# 先装依赖（torch/torchvision 已在基础镜像，requirements 里 torch>=2.1 会被判定为已满足，不重装）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 默认进 shell；具体动作用 Makefile 目标（make train/eval/export...）
CMD ["bash"]
