# 赛题#23 一键可运行环境(应对初赛"统一可运行性测试"硬门槛)
# 构建: docker build -t fintech23 .
# 自检: docker run --rm fintech23                 # 跑纯逻辑自测(无需 GPU/数据)
# 全流程: docker run --gpus all -v $PWD/data:/app/data fintech23 python embed.py --images ...
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app
COPY requirements.txt .
# torch 已在基础镜像;装其余依赖
RUN pip install --no-cache-dir -r requirements.txt || \
    pip install --no-cache-dir open_clip_torch timm faiss-cpu "numpy<2.0" pillow scikit-learn pandas matplotlib gradio tqdm

COPY . .

# 默认入口:跑全 10 模块自测(无需 GPU/数据,含**真实合成图像像素**的端到端去重),
# 供初赛"统一可运行性测试"一键确认环境 OK。features/synth_images/classify/embed/pipeline/viz_dedup
# 的经典 CPU 特征回退使分类→嵌入→去重→阈值全程在真实图像上跑通,不需 torch/GPU。
CMD ["python", "run_all_selftests.py", "--verbose"]
