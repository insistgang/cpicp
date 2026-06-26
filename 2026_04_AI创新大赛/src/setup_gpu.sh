#!/usr/bin/env bash
# =============================================================================
# setup_gpu.sh · GPU 机一键启用「真特征档」(华为 AOI few-shot)
# -----------------------------------------------------------------------------
# 在 GPU 算力机(2060+ / 或 Jetson Orin Nano)上一条命令:
#   ① 装 torch + timm + 项目依赖    ② 联网预缓存 timm 预训练权重(供离线复跑)
#   ③ 跑 feature_backend.py 自测     确认 TimmBackend 真后端可用、签名与经典后端一致
#   ④ 跑 run_real_pipeline.py        出真特征档的真实 AUC/F1/per-class/竞赛分
#   ⑤ 跑 bench_latency_gpu.py        在 2500×2500 上实测单图延时 vs <200ms@2060 红线
#
# 用法:
#   bash setup_gpu.sh                # x86 NVIDIA GPU,自动探测 CUDA 版本装匹配 torch
#   CUDA=cu118 bash setup_gpu.sh     # 手动指定 CUDA 轮子(cu121/cu118/cpu)
#   JETSON=1 bash setup_gpu.sh       # Jetson Orin Nano(aarch64,见下方 Jetson 分支)
#   MODEL=resnet18 bash setup_gpu.sh # 换更小 backbone(显存紧/求更快)
#
# 容错:set -e 任一步失败即停;关键命令均打印,失败处一目了然。
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")"                    # 切到 src/(脚本所在目录)
PY="${PY:-python3}"                     # 可用 PY=<解释器> 覆盖
MODEL="${MODEL:-wide_resnet50_2}"       # TimmBackend 默认 backbone
echo "============================================================"
echo "[0/5] 环境信息"
echo "  解释器 : $PY ($($PY --version 2>&1))"
echo "  目录   : $(pwd)"
$PY -c 'import sys; print("  pip    :", sys.prefix)'
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || echo "  (未检测到 nvidia-smi)"

# -----------------------------------------------------------------------------
# [1/5] 安装 torch + timm + 项目依赖
# -----------------------------------------------------------------------------
echo "============================================================"
echo "[1/5] 安装 torch / timm / 项目依赖"
$PY -m pip install --upgrade pip

if [ "${JETSON:-0}" = "1" ]; then
  # --- Jetson Orin Nano(aarch64,JetPack/L4T)分支 ---------------------------
  # Jetson 上 **不能** 用 PyPI 的 x86 torch 轮子。必须用 NVIDIA 官方 Jetson wheel:
  #   1) 查 JetPack 版本:  cat /etc/nv_tegra_release
  #   2) 到 NVIDIA 论坛 "PyTorch for Jetson" 置顶帖下载与 JetPack 匹配的
  #      torch-*-cp3X-linux_aarch64.whl,然后:
  #        pip install torch-*-linux_aarch64.whl
  #   或用 NVIDIA 提供的 l4t 容器(已内置 torch),在容器内跑本脚本(JETSON=1 跳过装 torch)。
  echo "  [Jetson] 跳过 pip 装 torch —— 请先按 NVIDIA 官方 Jetson wheel 装好 torch。"
  echo "          (查 JetPack: cat /etc/nv_tegra_release;下载置顶帖匹配 whl 后 pip install)"
  $PY -m pip install "timm>=0.9.12"
else
  # --- x86 NVIDIA GPU 分支 ----------------------------------------------------
  # 自动探测 CUDA 大版本选官方轮子;探测不到则回退 cu121(可用 CUDA=... 覆盖)。
  if [ -z "${CUDA:-}" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
      CUDA_VER=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]*\)\.\([0-9]*\).*/\1\2/p' | head -1)
      case "$CUDA_VER" in
        12*) CUDA=cu121 ;;
        118|117|116|11) CUDA=cu118 ;;
        *)   CUDA=cu121 ;;
      esac
    else
      CUDA=cu121
    fi
  fi
  echo "  CUDA 轮子: $CUDA  (可用 CUDA=cu118/cu121/cpu 覆盖)"
  $PY -m pip install torch --index-url "https://download.pytorch.org/whl/${CUDA}"
  $PY -m pip install "timm>=0.9.12"
fi

# 项目科学栈(干净环境补齐;已装则 pip 自动跳过)
$PY -m pip install -r requirements-gpu.txt || \
  $PY -m pip install numpy scipy scikit-learn Pillow matplotlib pandas pyyaml

echo "  torch / timm 版本:"
$PY -c 'import torch, timm; print("   torch", torch.__version__, "cuda?", torch.cuda.is_available(), "| timm", timm.__version__)'

# -----------------------------------------------------------------------------
# [2/5] 预缓存 timm 预训练权重(联网一次,之后可离线复跑)
# -----------------------------------------------------------------------------
echo "============================================================"
echo "[2/5] 预缓存预训练权重: $MODEL"
$PY - "$MODEL" <<'PYEOF'
import sys, timm, torch
name = sys.argv[1]
m = timm.create_model(name, pretrained=True, features_only=True, out_indices=(2, 3))
m.eval()
chs = m.feature_info.channels()
print(f"   ✅ 权重已缓存: {name}  layer2+layer3 通道={chs} → 拼接 D={sum(chs)}")
print(f"   缓存目录: ~/.cache/torch/hub 与 ~/.cache/huggingface")
PYEOF

# -----------------------------------------------------------------------------
# [3/5] feature_backend 自测 —— 确认真后端 + 回退契约都成立
# -----------------------------------------------------------------------------
echo "============================================================"
echo "[3/5] feature_backend.py 自测(经典后端必过 + TimmBackend 契约校验)"
$PY feature_backend.py

# 额外:GPU 机上直接验证 TimmBackend 真的能实例化并出对的形状(经典后端 D 之外)。
echo "  --- 真后端实跑校验(TimmBackend 出 (grid,grid,D) / (P,D)) ---"
MODEL="$MODEL" $PY - <<'PYEOF'
import os, numpy as np
from feature_backend import get_backend
be, is_real = get_backend(prefer_real=True, grid=8, model_name=os.environ.get("MODEL", "wide_resnet50_2"))
assert is_real, "❌ 期望 TimmBackend 真后端,却回退了经典后端(检查 torch/timm 是否装好)"
img = (np.random.RandomState(0).rand(2500, 2500, 3) * 255).astype(np.uint8)
pf = be.patch_features(img); ip = be.image_patches(img)
assert pf.shape == (8, 8, be.feat_dim), f"网格形状错: {pf.shape}"
assert ip.shape == (64, be.feat_dim), f"展平形状错: {ip.shape}"
norms = np.linalg.norm(ip, axis=1)
assert np.allclose(norms, 1.0, atol=1e-3), f"未 L2 归一化: 范数均值 {norms.mean():.3f}"
print(f"   ✅ TimmBackend 真后端: {be.name}  D={be.feat_dim}  patch_features={pf.shape}  image_patches={ip.shape}  L2≈1")
PYEOF

# -----------------------------------------------------------------------------
# [4/5] 真特征档端到端流水线 —— 真实 AUC/F1/per-class/竞赛分
# -----------------------------------------------------------------------------
echo "============================================================"
echo "[4/5] run_real_pipeline.py(真特征档:get_backend 自动切 TimmBackend)"
$PY run_real_pipeline.py --real
echo "  (报告落盘: ../output/pipeline_report.json,is_real_feature 应为 true)"

# -----------------------------------------------------------------------------
# [5/5] 2500×2500 单图延时 vs <200ms@2060 红线
# -----------------------------------------------------------------------------
echo "============================================================"
echo "[5/5] bench_latency_gpu.py(2500×2500 真特征延时,红线 <200ms@2060)"
$PY bench_latency_gpu.py --real --size 2500 --runs 20

echo "============================================================"
echo "✅ GPU 真特征档全部就绪。"
echo "   预期:第[3]步 is_real 真后端校验通过;第[4]步 is_real_feature=true 的真 AUC/F1;"
echo "        第[5]步 GPU 档单图 <200ms PASS(2060 级显卡)。若 [5] FAIL,可 MODEL=resnet18 换更小 backbone 复跑。"
