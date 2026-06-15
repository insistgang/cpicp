# C_baseline · 港股IPO风险预警 服务镜像（满足赛题"可运行性测试"）
# 默认 CMD 起 FastAPI 服务；大模型走云 API(.env)，本地不跑大模型。
FROM python:3.10-slim

WORKDIR /app

# poppler-utils: pdf2image/解析依赖; libgl/glib: OCR/视觉库运行库
RUN apt-get update && apt-get install -y --no-install-recommends \
        poppler-utils libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# 注: MinerU 会带入 torch 等较重依赖，首次构建较慢；如只需跑 API/评测可注释 mineru 行加速
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
# 可运行性 gate：容器启动即拉起预警 API 服务
CMD ["uvicorn", "api_app:app", "--host", "0.0.0.0", "--port", "8000"]
