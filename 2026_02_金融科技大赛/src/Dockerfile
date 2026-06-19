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

# 默认入口:跑可在无 GPU/无数据环境验证的纯逻辑自测(供可运行性测试快速确认环境 OK)
CMD ["sh", "-c", "python prepare_data.py --selftest && python metrics.py && python similarity.py && python app_demo.py --demo"]
