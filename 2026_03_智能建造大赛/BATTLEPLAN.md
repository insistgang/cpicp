# 03 智能建造 赛题7 · 冲国一技术作战方案

> 低空智能水上救援高精度视觉识别（命题方 深圳大江智造）。多角度联网调研后裁决。
> 一句话路线:**单帧 YOLOv12n+P2 锁召回 → TRT INT8+Super Mode+端到端FPS 守住30红线 → "检测框→GPS救援航点"系统闭环拉开国一差距。先焊死出局线,再讲创新。**

## 模型选型
- **主力 yolo12n + P2 四头**(已就位 `src/configs/yolo12-p2.yaml`,scale=n)。YOLOv12 的 area-attention 本身满足官方"注意力增强"要求,**不再额外叠 CBAM**。
- s 档仅作精度上界对照(证明测过),**不上端侧**——n→s 在 Orin 上掉的 FPS 比"640→768"更贵。
- 精度缺口用 **P2(stride-4,小目标最大单一杠杆)+ 4070S 长训 + 增广**补回。
- **必做消融**:四头 vs 去 P5 三头(海面几乎无大目标,P5 可能冗余,砍掉换帧率)。端侧规模/分辨率以 `trt_infer_orin.py` 端到端实测拐点定,不靠直觉。

## 优化栈(按性价比排序,单兵能落地为准)
1. **TRT INT8 混合精度**(敏感层回退FP16)+ **真实海况帧校准**(300-500张反光/波纹/运动模糊帧,禁用干净val图;掉>2% AP_small 退FP16)— 与 P2 同等紧急,决定是否出局
2. **Super Mode**(JetPack6.2 `nvpmodel -m 2` MAXN_SUPER 25W + `jetson_clocks`)— 零成本 1.7x 吞吐,DC 供电
3. **输入分辨率管控**:训练 1024(召回)、端侧导 640/768 两档 engine 按现场 FPS 余量选(640→768 像素翻倍、延迟同比涨,最大隐患)
4. **P2 头保留 + 四头/去P5 消融**
5. **NWD/Wasserstein 标签分配**(零推理成本,改 bbox_iou 加 wasserstein 分支,nwd_ratio≈0.5,A/B 验证)
6. **Wise-IoU/Inner-IoU 回归损失**(免费快赢)
7. **训练期物理增广**:Albumentations RandomSunFlare/Shadow/Fog/CLAHE + **自研 GT-Anchored Glint**(非GT水面区贴高斯高光斑当难负样本,直击反光误检)
8. **难负样本 + 空标签负图**(纯水面/碎浪/泡沫帧写空txt,占10-15%)— 救援场景 recall 优先
9. **POSEIDON 元数据 copy-paste 补浮标/船稀缺类**(离线)

**不上**:BiFPN(与现有 FPN+PAN 重叠)、在线超分/SAHI 实时切片(与30FPS死冲,只作离线高召回对照档)、多帧时序检测网络、改 QGC 源码自写 AI 插件(单兵被坑死)。

## ≥30FPS 方案 + 兜底降级链
**主配置**:yolo12n + P2 + 输入640(紧张升768)+ TRT INT8 混合精度 + Super Mode + NMS 留 GPU(EfficientNMS plugin 或 NMS-free 头)+ 纯 TRT+OpenCV 流水线。
参照实测:YOLOv8n INT8@640 标准模式 ~43FPS,叠 Super Mode 1.7x 有余量吸收 P2+后处理。
- **致命坑**:Orin Nano 这一 SKU **无硬件 NVENC 编码器**,RTSP 软编码 x264enc 吃多核 CPU → 优先 H265/低码率,**报含编码的端到端 FPS**,并备录屏轨。
- **必须**:`trt_infer_orin.py` 把解码+NMS+前处理全计入端到端计时(原为 TODO,裸推理 FPS 虚高),报「裸推理/含画框/含编码」三档。
- **降级链**(按序砍,精度优先级低→高):768→640 → 关P2/砍P5 → INT8替FP16 → 确认 Super Mode+DC 供电 → 抽帧推理(每帧必检测跳显示)→ 最后才动模型(n已最小)。每档 FPS×PR 留档,既兜底又是答辩弹药。

## 三个创新点(答辩 + 技术方案)
1. **系统创新(国三→国一分水岭):检测框→GPS救援航点闭环。** pymavlink 读 GLOBAL_POSITION_INT+ATTITUDE+高度,针孔模型把框中心射线与海平面(z=0)求交得目标经纬度,框旁标注 类别+经纬度+距本艇距离。用已知 GPS 的浮标当真值事后校验,**诚实报投影误差区间(几十米级,别吹米级)**。正是大疆系命题方"发现落水者并给可航行坐标"的业务语言。
2. **算法创新:水面小目标"反光误报抑制"链** = P2 + YOLOv12 area-attention + NWD标签分配 + GT-Anchored Glint 难负样本,配精度-速度双档(实时单帧 vs 离线SAHI高召回),两档 FPS+PR 都报,消融表论证每项独立增益。
3. **部署创新:Orin Nano 无硬编码器约束下的端到端实时管线工程优化。** 坦诚讲清 INT8 对小目标的取舍、软编码 H265 码率权衡、Super Mode 功耗档——把"为什么检测在边缘、地面站只可视化"答成"边缘计算+低延迟的正确架构",工程严谨性即综合表现力得分点。

## 交付物 → 评分维度映射
| 交付物 | 主要拿分维度 | 怎么拿分 |
|---|---|---|
| 技术方案文档 | 技术创新性+完成度 | 开篇放"硬指标达标对照表"先焊出局线;每个选择给"为什么"对上赛题约束;按官方四要点(选型/小目标特征/注意力/轻量化)分四章 |
| 源码(Python) | 完成度+综合表现力 | 一条命令跑通 train→eval→export→Orin测速;README 写清 Super Mode/供电/nvpmodel 档让评委可复现 FPS |
| 性能报告 | 完成度(30FPS唯一可信证明) | 各类PR曲线 + 分桶召回 + FP32/FP16/INT8×640/768 的 FPS-召回帕累托 + tegrastats 功耗显存;**报含画框+编码的端到端FPS** |
| QGC融合demo(加分) | 综合表现力+创新性 | GStreamer 闭环(appsink→TRT→OpenCV→appsrc→RTSP,QGC配RTSP Low Latency)+ 叠 MAVLink 遥测+AI框+落水告警;**双轨:主轨现场实时+保底录屏轨** |

## 周计划(集中编码 7/14–8/25,提交 8/31)
| 周 | 里程碑 | 交付 |
|---|---|---|
| 现在–7/13(Mac) | 代码/文档增量就绪 + 数据申请发起 | 改 export/trt(端到端三档+INT8)、加 NWD/WIoU、add_negatives、Albumentations+Glint、新建 geolocate/stream_qgc/时序滤波;方案+报告模板;**立即申请命题方数据** |
| 7/14–7/20 | 公开数据全链路+保底闭环 | SeaDronesSee+AFO train→eval→export;Orin 开 Super Mode 测 FP16 端到端基线;GStreamer→RTSP→QGC 第一周打通(保底) |
| 7/21–7/27 | **30FPS 红线焊死** | INT8+真实帧校准;FP16/INT8×640/768 端到端FPS+分桶召回矩阵;四头/去P5 消融定端侧配置 |
| 7/28–8/3 | 算法拉分 | NWD/WIoU/增广/难负样本/POSEIDON 逐项 A/B 消融表;最优权重重测端侧守线 |
| 8/4–8/10 | 系统创新 | geolocate 接真机 MAVLink,框→GPS,浮标真值校验+误差区间;ByteTrack 时序滤波降误报 |
| 8/11–8/17 | 域适配 | 伪标签 UDA 微调(公开→命题方),held-out 验证;核对类别口径;有余量上 DeepStream 做 QGC 融合加分 |
| 8/18–8/25 | 收尾打磨 | 三交付物定稿+路演双轨(录屏保底+现场实时+落水救援剧本演练) |
| 8/26–8/31 | 提交 | 技术方案+源码+性能报告+QGC demo 打包 |

## 头部风险
1. **FPS 是出局线不是加分项**:LAF-YOLOv10 P2 在 Orin Nano 实测仅 24.3FPS<30。现脚本只测裸推理,含画框+软编码必跌——先把整链计入再报数,否则路演翻车直接出局。
2. **Orin Nano 无硬件 NVENC**:软编码 x264enc 吃 CPU,裸推理45FPS 一开推流可能跌破30 → H265/低码率+jetson_clocks+报含编码端到端+录屏保底。
3. **INT8 对小目标敏感 + P2 高分辨率层放大量化误差**:每改一处(P2/分辨率/INT8)都重跑分桶召回,掉>2% 退 FP16,禁止只看总 mAP。
4. **数据需申请且基于 SeaDronesSee 二开有域差**:现在不申请会卡死后期;命题方类别口径(穿救生衣落水者是否独立判定)可能与 CLASS_MAP 错位,拿到第一件事就核对。
5. **单兵贪多必崩**:DeepStream/多帧时序网/改QGC源码都是时间黑洞 → 严守"先纯TRT+OpenCV保底闭环,创新留主线跑通后"。
6. **消融纪律**:每项增益独立 A/B,靠消融表而非直觉;报告写"参考xx思路"而非照搬论文百分比。
7. **recall 优先于 precision**:难负样本 10-15% 起,加太多(>30%)模型变保守漏检真人;用 PR 曲线找平衡。
