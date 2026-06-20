# Jetson Orin Nano 8G 端到端部署清单（小队照着刷）

> 赛题7「低空智能水上救援高精度视觉识别」的**端侧部署 runbook**。
> 03 是六赛里唯一给**奖金 + 实习**的，而 **动态视频 ≥30FPS 是硬出局线**——这份清单就是把
> 「需 GPU/Nano」的瓶颈拆成小队照着做的步骤，目标是阶段3 一刀砍出「过/不过」，越早越好。
>
> 本文所有 `python` 命令的 flag 均**已对照 `trt_infer_orin.py` / `stream_qgc.py` / `export_onnx.py`
> 的真实 argparse 核对一致**（核对结果见文末附录）。命令里出现的参数都真实存在，别凭记忆改名。
>
> 适用硬件：**Jetson Orin Nano 8G**（关键约束：此 SKU **无硬件 NVENC**，实战 RTSP 走软编码
> x264/x265 吃 CPU，含编码档会进一步降；TensorRT 由 JetPack 自带；需 `pip install pycuda opencv-python`）。
> 训练/导出 ONNX 在 RTX4070S 上做（见 `README.md` 的 runbook），**本清单只覆盖 Orin 端**。

---

## 总览：5 个阶段

| 阶段 | 做什么 | 产出/判定 | 卡住了看 |
|---|---|---|---|
| 0 刷机 | JetPack 烧录 + 首启 + 功耗拉满 | `nvpmodel`/`jetson_clocks` 生效 | 踩坑①② |
| 1 装环境 | pycuda + opencv + 确认 TRT 版本 | `import tensorrt, pycuda` 不报错 | 踩坑③ |
| 2 建引擎 | trtexec 建 FP16 / INT8 引擎 | `best_fp16.engine` 落盘 | 踩坑④⑥ |
| 3 测速 ★第一刀★ | 三档测速 vs 30FPS | **③含编码 PASS/FAIL** | 决策树 |
| 4 实时闭环 | RTSP→QGC + 录屏保底 | QGC 出画 + `out.mp4` 存证 | 踩坑⑤ |

> **强烈建议先跑阶段3 拿到「③含编码」FPS，再回头精调精度。** 30FPS 过不了，精度再高也出局。

---

## 阶段0 · 刷机与功耗（一次性）

### 0.1 选 JetPack 版本
- 推荐 **JetPack 6.x（L4T r36.x）**：自带 **TensorRT 10.x**、CUDA 12、cuDNN，与 `trt_infer_orin.py`
  的 TRT10 新 API（`num_io_tensors` / `execute_async_v3`）匹配。
- 若手头是旧卡刷的 JetPack 5.x（TensorRT 8.x）也能跑——脚本**已同时兼容 TRT8/TRT10**，自动探测，
  无需改代码。只是 8.x 的 INT8/算子支持略旧，优先 6.x。
- 版本一旦定下，**全队统一**：引擎文件不跨 TRT 大版本通用（见踩坑⑥）。

### 0.2 烧录
1. 用 **NVIDIA SDK Manager**（一台 Ubuntu 主机）或官方 SD 卡镜像烧录。Orin Nano 开发套件走 SD 卡最省事；
   有 NVMe 的话 SDK Manager 刷到 NVMe 更快更稳。
2. 进入 recovery（跳线/按键按板子手册），SDK Manager 选对应 JetPack，勾上 **Jetson SDK Components**
   （CUDA / TensorRT / cuDNN / OpenCV），一并装上。
3. 首次启动：完成 Ubuntu OOBE（用户名/时区/网络）。**记下板子 IP**（阶段4 QGC 要用）：`ip addr`。

### 0.3 功耗模式拉满（直接影响 FPS，必做）
Orin Nano 默认可能在低功耗档，FPS 会虚低。测速前务必拉满：
```bash
# 查看当前功耗模式
sudo nvpmodel -q
# 切到最高功耗档（Orin Nano 8G: MAXN / 模式 0；具体编号以 nvpmodel -q 列出的为准）
sudo nvpmodel -m 0
# 锁定 GPU/CPU/EMC 时钟到最高（关闭动态降频，测速稳定）
sudo jetson_clocks
# 确认时钟已拉满
sudo jetson_clocks --show | head
```
> 报告里的 FPS 必须是在 `nvpmodel -m 0 + jetson_clocks` 下测的，否则数字不可复现。
> 答辩可被问「功耗模式是哪档」——记得截图 `nvpmodel -q` 和 `tegrastats`。

### 0.4 装监控工具（旁路记功耗/温度）
```bash
sudo pip3 install -U jetson-stats   # 提供 jtop
sudo systemctl restart jtop.service
jtop                                 # 实时看 GPU 占用/功耗/温度
# 或命令行旁路记录（测速时另开一个终端）：
tegrastats --interval 1000
```

---

## 阶段1 · 装环境

### 1.1 确认 TensorRT 已随 JetPack 装好
```bash
# trtexec 路径（建引擎用，阶段2 反复用）
ls -l /usr/src/tensorrt/bin/trtexec
# 确认 Python 能 import tensorrt 并打印版本（决定脚本走 TRT8 还是 TRT10 分支）
python3 -c "import tensorrt as trt; print('TensorRT', trt.__version__)"
```
- 打印 `10.x` → 脚本走新 API；`8.x` → 走旧 API。**都支持**，记一下版本，引擎别跨版本拷。

### 1.2 装 pycuda + opencv（脚本硬依赖）
```bash
# pycuda：trt_infer_orin.py 的 H2D/D2H/显存分配靠它
sudo apt-get update && sudo apt-get install -y python3-dev build-essential
pip3 install --user pycuda

# OpenCV：preprocess / 画框 / 编码 / GStreamer 推流都要
# 优先用 JetPack 自带的带 GStreamer 的 OpenCV（SDK Manager 勾了就有）；
# 没有再 pip 装（pip 版可能不带 GStreamer 后端，会影响阶段4 RTSP 推流）
python3 -c "import cv2; print('OpenCV', cv2.__version__)" || pip3 install --user opencv-python
```
> **GStreamer 后端是阶段4 推流的前提**。验证：
> ```bash
> python3 stream_qgc.py --selftest
> ```
> 看输出里 `OpenCV ... GStreamer 后端: YES`。若显示「未启用」，阶段4 的 RTSP 推流会失败——
> 改用 JetPack 自带 OpenCV，或自行编译带 `-D WITH_GSTREAMER=ON` 的 OpenCV。
> （本机 Mac 上跑这条会显示 `OpenCV 未安装` + 打印将用的推流管线，属正常自检；真跑须上 Orin。）

### 1.3 装项目依赖（PC 端依赖里 Orin 不需要 torch/onnxruntime）
ONNX 是在 PC（RTX4070S）上由 `export_onnx.py` 导好后**拷贝到 Orin** 的，Orin 端**不需要** torch /
ultralytics / onnxruntime。Orin 只需要：
```bash
# 已在 1.2 装了 pycuda + opencv；再补 numpy（preprocess/postprocess 用）和 pymavlink（阶段4 接遥测，可选）
pip3 install --user "numpy<2.0" pymavlink
```
> `requirements.txt` 顶部那批（torch/ultralytics/onnx…）是**训练/导出端**装的，**不要在 Orin 上装**
> （文件末尾已注明）。Orin 端的 torch 若真要装得用 NVIDIA Jetson 专版 wheel，但本部署链路用不到。

---

## 阶段2 · 建 TensorRT 引擎

前置：已把 PC 导好的 ONNX 拷到 Orin。**端侧实时务必用 640/768 两档**（`imgsz=1024` 是训练/精度档，
端侧跑不到 30FPS）。PC 端导出命令（在 RTX4070S 上，仅供对照，flag 见 `export_onnx.py`）：
```bash
# PC 端：导 640 / 768 两档
python export_onnx.py --weights runs/detect/train/weights/best.pt --imgsz 640
python export_onnx.py --weights runs/detect/train/weights/best.pt --imgsz 768
```

### 2A · FP16 引擎（端侧默认，先用这个过阶段3）
用 JetPack 自带 trtexec 在 Orin 上原生建：
```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=best.onnx \
  --saveEngine=best_fp16.engine \
  --fp16
```
> INT8 对小目标敏感，**端侧默认 FP16**；FP16 跑得过 30FPS 就别上 INT8（省一轮精度回归）。

### 2B · INT8 引擎（FP16 不够快时才上，需真实海况帧校准）
INT8 校准帧要用**命题方二开数据集里真实的反光/波纹/运动模糊帧**（300–500 张），别用合成帧，
否则量化标定偏，落水小目标会漏检。

**法一（推荐）· trtexec 直接建**（最省事，与脚本 docstring 一致）：
```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=best.onnx \
  --saveEngine=best_int8.engine \
  --int8 --fp16            # 混合精度：敏感层回退 FP16，保小目标精度
```

**法二 · 用脚本内置 EntropyCalibrator2 建**（要喂校准图目录，落 `int8_calib.cache`）：
```bash
# 校准图目录：放 300-500 张真实海况帧（反光/波纹/运动模糊）
python3 trt_infer_orin.py \
  --onnx best.onnx \
  --int8 \
  --calib-dir ./calib_frames \
  --imgsz 640 \
  --benchmark
```
> `--int8` **必须配 `--calib-dir`**（脚本会报错提醒）。法二建完直接接着测速，一步到位。
> 注意：INT8 引擎上线前**必做精度回归**（在 PC 上对比 INT8 vs FP16 的分桶召回，别让 30FPS 把人命漏了）。

---

## 阶段3 · 测速（★第一刀·决定整盘方案生死★）

`trt_infer_orin.py` 做**真实端到端三档计时**，每档与 30FPS 比较并打 PASS/FAIL：
- **① 裸推理** = H2D + GPU 推理 + D2H + sync（会虚高，别拿这个报）
- **② 含后处理** = ① + Detect 解码 + NMS + 画框
- **③ 含编码（实战代理）** = ② + 帧编码（JPEG 代理；真实 RTSP/H265 见阶段4，更重）

**报告以「③ 含编码」为准。**

### 3.0 先在本机/Orin 焊死解码逻辑（纯 numpy，无需引擎）
```bash
python3 trt_infer_orin.py --selftest
```
真实输出（Mac 与 Orin 一致，纯 numpy）：
```
  OK  NMS:重叠框留高分(keep=[0])
  OK  NMS:不重叠框全保留
  OK  NMS:空输入返回空(不崩)
  OK  病态极小张量(锚点<列数)→ 返回空(护栏,不抛 argmax 异常)

OK trt_infer_orin 解码自测通过
```
> 这条焊死 postprocess 布局判定 / xywh→xyxy / conf 过滤 / NMS（10 项）。现场跑错会让所有框错位，
> 先本地过了再上引擎。

### 3.1 三档测速命令（在 Orin 上跑）
```bash
# 用 FP16 引擎 + 真实视频测速（imgsz 与导出/建引擎一致！这里 640）
python3 trt_infer_orin.py \
  --engine best_fp16.engine \
  --imgsz 640 \
  --source test_sea.mp4 \
  --benchmark
```
- **`--source` 强烈建议给真实海况视频**：不给会用合成帧，后处理框数随机，③ 档不真实（脚本会 `[warn]`）。
- 也可从 ONNX 现建现测（省一步落盘）：
  ```bash
  python3 trt_infer_orin.py --onnx best.onnx --fp16 --imgsz 640 --source test_sea.mp4 --benchmark
  ```
- 升 768 档对照（小目标紧张时）：把上面两处 `640` 换成 `768`，并用 768 的引擎/ONNX。

测速输出长这样（FPS 是真机实测，本机无 GPU 跑不出数字，仅示意格式）：
```
=== Orin Nano 端到端测速 (imgsz=640, iters=200) ===
档位                              时延ms      FPS   vs 30FPS红线
① 裸推理                          x.xx     xx.x   ✅ PASS / ❌ FAIL
② 含后处理(解码+NMS+画框)          x.xx     xx.x   ✅ PASS / ❌ FAIL
③ 含编码(实战代理)                 x.xx     xx.x   ✅ PASS / ❌ FAIL   ← 看这行
  显存: xxxx / 7xxx MB
```

### 3.2 「过 / 不过」决策树
```
跑 ③ 含编码档 FPS：
├─ ③ ≥ 30  → ✅ 过线。锁定该 (imgsz, 精度档) 为基线，进阶段4 用 stream_qgc 实测含编码端到端
│            （RTSP/H265 软编更重，③可能再降一截，以阶段4 为最终数）。先别贪精度回去加 P2/升分辨率把线压崩。
│
└─ ③ < 30  → ❌ 启动降级链（按"省得多、伤精度少"排序，逐项试，每加一项重测 ③）：
   1) 768 → 640        分辨率降档（最立竿见影；小目标召回会掉，配阶段2 的真实帧验证）
   2) 关 P2 头          P2(stride4) 算力贵；端侧关掉换回标准头，FPS 涨明显
   3) yolo12 → yolo12n  换更轻骨干（n 档），重新导 ONNX→建引擎→重测
   4) FP16 → INT8       上 INT8（阶段2B），需真实海况帧校准 + 精度回归
   5) Super Mode        JetPack 6 的 Orin Nano Super 固件可提算力，确认已刷到 Super 档（nvpmodel 看是否有更高档）
   6) 抽帧推理          每 N 帧推一次，中间帧用 track_filter 时序预测补（端到端"等效帧率"达 30，最后兜底）
   逐项叠加直到 ③ ≥ 30。每改一档都重跑 3.1，并记下该档的精度代价（分桶召回）。
```

---

## 阶段4 · 实时闭环（RTSP → QGroundControl + 录屏保底）

`stream_qgc.py` 跑实战含编码端到端：采集 → TRT 推理 → 时序滤波 → 画框+落水告警+检测框→GPS+遥测条 →
**H265 软编码（Orin Nano 无 NVENC，用 x265enc）** → RTSP → QGC。

### 4.0 先自检接线（Mac 上也能跑，确认管线字符串/GStreamer/模块接线）
```bash
python3 stream_qgc.py --selftest
```
真实输出（Mac 上 cv2 缺失属正常，Orin 上应显示 GStreamer: YES）：
```
  ✅ track_filter / geolocate 模块可导入(接线正常)
  ⚠️  OpenCV 未安装(Orin 上必装): No module named 'cv2'   ← Orin 上应为 GStreamer 后端: YES

  将使用的输出管线(Orin Nano 软编码 H265 → RTSP):
    appsrc ! videoconvert ! x265enc tune=zerolatency bitrate=4000 speed-preset=ultrafast ! rtph265pay config-interval=1 pt=96 ! udpsink host=127.0.0.1 port=5600
  QGC 配置:Application Settings → Video → Source=RTSP,URL=rtsp://<orin-ip>:8554/...,开 Low Latency

✅ stream_qgc 自检通过(骨架就绪,真跑需上 Orin)
```

### 4.1 真跑：推理 + 推流 + 同时录屏存证（双轨保底）
```bash
python3 stream_qgc.py \
  --engine best_fp16.engine \
  --imgsz 640 \
  --source test_sea.mp4 \
  --width 1280 --height 720 \
  --host <地面站IP> --port 5600 \
  --bitrate 4000 \
  --record out.mp4
```
- `--source` 留空则用 `videotestsrc` 跑通管线（先验证推流链路，再换真视频/相机）。
- `--record out.mp4` = **录屏保底轨**：现场任一环节卡顿，切录屏轨，本地 `out.mp4` 同时存证。
- 接 MAVLink 遥测（检测框→GPS 航点、顶部遥测条）时加：`--mavlink udp:127.0.0.1:14550`
  （`--hfov 84.0` 是默认水平视场角，按你相机改）。
- 运行时每 30 帧打印一次 `推流中 N 帧, 端到端 X.X FPS` —— **这个 X.X 才是报告里"实战含编码端到端 FPS"**，
  比阶段3 的 ③ JPEG 代理更接近真值（软 H265 更重）。同时另开终端 `tegrastats` 记功耗。

### 4.2 QGC 侧配置
- QGroundControl → Application Settings → Video → **Source = RTSP**，
  URL 填 `rtsp://<orin-ip>:8554/...`（或按你的 RTSP server 地址），勾 **Low Latency**。
- 若用上面 `udpsink`（UDP/RTP）而非 RTSP server，QGC 可改用 UDP h265 源对 `--port 5600`；
  要标准 RTSP URL 则在 Orin 上配 `rtsp-simple-server`/`mediamtx` 把 UDP 转 RTSP。
- **保底**：现场网络抖动就切 `--record` 的 `out.mp4` 放录屏，别让答辩黑屏。

---

## 踩坑清单

| # | 现象 | 原因 | 处理 |
|---|---|---|---|
| ① | FPS 比预期低一截、不稳定 | 没拉功耗模式/时钟，动态降频 | `sudo nvpmodel -m 0 && sudo jetson_clocks`，测速全程保持 |
| ② | 跑一会儿掉频/降速 | 散热不足，温度墙 | 装风扇/散热片，`jtop` 看温度，别在闷箱里测 |
| ③ | `pip install pycuda` 失败 | 缺 `python3-dev`/编译器，或没 CUDA 环境变量 | `apt install python3-dev build-essential`；确认 `nvcc -V` 能找到 CUDA（`export PATH=/usr/local/cuda/bin:$PATH`） |
| ④ | 引擎构建失败 / 显存不足 OOM | 8G 显存吃紧，workspace 太大或 imgsz 太大 | 降 imgsz（768→640）、关其他占显存进程、INT8 减显存；脚本内 workspace 已限 1GB |
| ⑤ | RTSP 推流不出画 / `VideoWriter` 打不开 | OpenCV 没带 GStreamer 后端 | `stream_qgc.py --selftest` 看 GStreamer YES/NO；用 JetPack 自带 OpenCV 或重编带 GStreamer |
| ⑥ | 加载 `.engine` 报反序列化失败 | 引擎跨 TRT 大版本/跨设备不通用 | **在该 Orin 上、该 JetPack 的 TRT 版本下重新 trtexec 建引擎**，别拷别人的 engine |
| ⑦ | 含编码档 ③ 远低于 ②（编码占大头） | Orin Nano **无硬件 NVENC**，x265 软编吃 CPU | 降 `--bitrate`、`speed-preset=ultrafast`（已默认）、降分辨率；报告注明软编码瓶颈 |
| ⑧ | 测速框数乱跳、③ 不真实 | 没给 `--source`，用了合成帧 | 测速必带 `--source 真实海况视频` |
| ⑨ | 框全部错位 | 后处理布局判错（极罕见） | 先 `trt_infer_orin.py --selftest` 焊死解码；确认 ONNX 是无 NMS 导出（与 postprocess 匹配） |

---

## 附录 · flag 核对结果（已对照源码 argparse）

> 下表确认本文每条 `python` 命令的 flag 都真实存在于对应脚本。核对方法：
> `grep -n "add_argument" trt_infer_orin.py stream_qgc.py export_onnx.py`。

**`trt_infer_orin.py`**（`main()` argparse）：
`--selftest` `--onnx` `--engine` `--fp16` `--int8` `--calib-dir` `--source` `--imgsz`(默认640) `--benchmark`
→ 本文用到的 `--engine/--onnx/--fp16/--int8/--calib-dir/--source/--imgsz/--benchmark/--selftest` **全部命中**。
约束：无 `--onnx` 也无 `--engine` 时报错 `需 --onnx 或 --engine 之一`；`--int8` 须配 `--calib-dir`（已在 2B 标注）。

**`stream_qgc.py`**（`main()` argparse）：
`--selftest` `--onnx` `--engine` `--fp16` `--int8` `--calib-dir` `--source` `--mavlink` `--imgsz`(默认640)
`--width`(默认1280) `--height`(默认720) `--hfov`(默认84.0) `--host`(默认127.0.0.1) `--port`(默认5600)
`--bitrate`(默认4000) `--record`
→ 本文用到的 `--engine/--imgsz/--source/--width/--height/--host/--port/--bitrate/--record/--mavlink/--hfov/--selftest` **全部命中**。

**`export_onnx.py`**（PC 端，对照用）：
`--weights`(required) `--imgsz`(默认1024) `--opset`(默认12) `--simplify`/`--no-simplify` `--nms`
→ 本文用到的 `--weights/--imgsz` **命中**。

**软编码管线**（`stream_qgc.py` 的 `GST_OUT` 常量，自检实跑打印一致）：
`appsrc ! videoconvert ! x265enc tune=zerolatency bitrate=4000 speed-preset=ultrafast ! rtph265pay config-interval=1 pt=96 ! udpsink host=127.0.0.1 port=5600`

**`trtexec`** 路径与 flag 与 `trt_infer_orin.py` docstring 一致：
`/usr/src/tensorrt/bin/trtexec --onnx=... --saveEngine=... --fp16`（或 `--int8 --fp16`）。

---

## 仍需在 GPU / Nano 上才能验证的部分（本机 Mac 无法跑出）

- 阶段2 建引擎（trtexec / 脚本内 INT8 校准）：需 TensorRT + pycuda + 真实 ONNX/校准帧。
- 阶段3 三档 FPS 数字与 PASS/FAIL：需 Orin GPU + 引擎 + 真实视频；本机只能跑 `--selftest`（纯 numpy 解码，已过）。
- 阶段4 RTSP→QGC 实时画面、`--record` 存证、`stream_qgc` 的端到端 FPS：需 Orin + 带 GStreamer 的 OpenCV +（可选）MAVLink/QGC。
- 功耗/温度/显存实测（`tegrastats`/`jtop`/`nvpmodel`）：需真机。

本机已验证：解码逻辑自测、模块接线自检、软编码管线字符串、所有 flag 与源码 argparse 一致。
