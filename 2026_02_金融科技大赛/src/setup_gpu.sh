#!/usr/bin/env bash
# =============================================================================
# 赛题#23 一键上机包(算力机 / GPU) — 装 torch+open_clip → 真特征跑 classify/embed
#   → pipeline 用 CLIP 真特征出真指标。与 04 的 run_all_selftests.sh 同风格。
#
# 本机 Mac(无 torch/GPU)只跑 baseline:`python run_all_selftests.py` 即可，本脚本无需跑。
# 小队在算力机上照着跑:
#   bash setup_gpu.sh                 # 默认 CPU-baseline 自检 + 提示装 GPU 栈
#   INSTALL=1 bash setup_gpu.sh       # 真装 torch+open_clip+timm+faiss 再跑 CLIP 真特征
#   CUDA=cu121 INSTALL=1 bash setup_gpu.sh   # 指定 CUDA wheel(默认 cu121，按算力机改)
# Jetson Orin Nano:torch 须用 NVIDIA 官方 JetPack wheel(见文末「Nano 验证」)。
# =============================================================================
set -e
cd "$(dirname "$0")"

# ---- 选一个带科学栈的 python3(同 04 的 pick_py;可用 PY=<解释器> 覆盖) ----
pick_py() {
  for c in "$PY" python3 /usr/bin/python3 python; do
    [ -n "$c" ] || continue
    if "$c" -c 'import numpy' >/dev/null 2>&1; then echo "$c"; return 0; fi
  done
  echo "❌ 找不到带 numpy 的 python3(请装 numpy 或用 PY=<解释器> 指定)" >&2
  return 1
}
PYBIN="$(pick_py)"
echo "解释器: $PYBIN ($("$PYBIN" --version 2>&1))"

CUDA="${CUDA:-cu121}"          # torch CUDA wheel 标签;按算力机改(cu118/cu121/cu124)
OUT="${OUT:-output/gpu_run}"   # 真特征端到端输出目录
NG="${NG:-40}"                 # 合成影像组数
RF="${RF:-0.35}"              # 套用/重复比例

# ---------------------------------------------------------------------------
# 步骤 1:安装 GPU 真特征栈(仅 INSTALL=1 时执行,避免误装)
# ---------------------------------------------------------------------------
if [ "${INSTALL:-0}" = "1" ]; then
  echo "=== [1/4] 安装 GPU 真特征栈 (CUDA=$CUDA) ==="
  # torch/torchvision 走官方 CUDA index;其余从 requirements 的 GPU 区块装
  "$PYBIN" -m pip install --upgrade pip
  "$PYBIN" -m pip install "torch>=2.1" "torchvision>=0.16" \
      --index-url "https://download.pytorch.org/whl/${CUDA}"
  "$PYBIN" -m pip install "open_clip_torch>=2.24" "timm>=0.9" "faiss-cpu>=1.8" \
      "numpy<2.0" "pillow>=10.0" "scikit-learn>=1.3" "pandas>=2.0" \
      "matplotlib>=3.7" "gradio>=4.0" "tqdm>=4.66"
else
  echo "=== [1/4] 跳过安装(INSTALL=1 才装 torch+open_clip);仅做 CPU baseline 自检 ==="
fi

# ---------------------------------------------------------------------------
# 步骤 2:确认 torch/CLIP 是否就绪(决定走真特征还是 baseline)
# ---------------------------------------------------------------------------
echo "=== [2/4] 探测 torch / open_clip / CUDA ==="
HAS_TORCH=0
if "$PYBIN" -c 'import torch, open_clip' >/dev/null 2>&1; then
  HAS_TORCH=1
  "$PYBIN" - <<'PY'
import torch
print(f"  torch {torch.__version__} | CUDA 可用: {torch.cuda.is_available()}"
      + (f" | GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else " (CPU-only torch)"))
PY
else
  echo "  ⚠️ 未检测到 torch+open_clip → 本次只能跑 CPU baseline(经典特征)。"
  echo "     算力机请用 INSTALL=1 bash setup_gpu.sh 先装 GPU 栈。"
fi

# ---------------------------------------------------------------------------
# 步骤 3:生成合成影像 → classify/embed 真特征自检
# ---------------------------------------------------------------------------
echo "=== [3/4] 生成合成影像 + 模块自检 ==="
"$PYBIN" synth_images.py --out "$OUT" --n-groups "$NG" --reuse-frac "$RF"

if [ "$HAS_TORCH" = "1" ]; then
  echo "--- classify.py: CLIP 零样本筛面签(真特征) ---"
  "$PYBIN" classify.py --mode zeroshot --images "$OUT"/img_*.png
  echo "--- embed.py: CLIP 真特征嵌入(--backend clip) ---"
  "$PYBIN" embed.py --images "$OUT"/img_*.png --backbone clip --backend clip --out "$OUT/emb"
else
  echo "--- (baseline) classify/embed 走经典特征自测 ---"
  "$PYBIN" classify.py --selftest
  "$PYBIN" embed.py --selftest
fi

# ---------------------------------------------------------------------------
# 步骤 4:pipeline 端到端出真指标
#   有 torch → --backend clip(CLIP 真特征);无 torch → --backend classic(baseline)
# ---------------------------------------------------------------------------
echo "=== [4/4] pipeline 端到端真指标 ==="
if [ "$HAS_TORCH" = "1" ]; then
  "$PYBIN" pipeline.py --real-images "$OUT" --backend clip --clip-backbone clip --plot
  echo "✅ CLIP 真特征端到端完成 → $OUT(含 dedup_viz.png + 真 AUC/Top-k/最优阈值)"
else
  "$PYBIN" pipeline.py --real-images "$OUT" --backend classic --plot
  echo "✅ CPU baseline 端到端完成 → $OUT(经典特征真指标;装 torch 后改 --backend clip 切 CLIP)"
fi

cat <<'NOTE'

────────────────────────────────────────────────────────────────────────────
【GPU 上机验证清单】(本机 Mac 无 torch 跑不到真特征分支,以下须在算力机实测)
 1. INSTALL=1 bash setup_gpu.sh          # 装 torch+open_clip+timm+faiss
 2. 看步骤[2/4]打印 "CUDA 可用: True"      # 确认 GPU 被吃到
 3. 步骤[3/4] embed.py 不再打印"回退经典特征" # 说明走的是 CLIP 真特征
 4. 步骤[4/4] [3/6] 行显示 "CLIP/clip 真特征提取"(而非"经典特征") # 真特征落地
 5. 比对 AUC/最优阈值与 baseline:CLIP 真特征对"细微套用"对的判别应更优
【Jetson Orin Nano】torch 不能 pip 普通 wheel,须用 NVIDIA JetPack 对应版本:
   参见 forums.developer.nvidia.com 的 PyTorch for Jetson;装好后 CUDA=（留空，用Jetson自带）
   INSTALL=0 bash setup_gpu.sh(手动装 torch 后),其余步骤同上。
────────────────────────────────────────────────────────────────────────────
NOTE
