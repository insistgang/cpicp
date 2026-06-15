# 赛事 E · 第三届中国研究生操作系统开源创新大赛（暨开放原子大赛操作系统专项赛）
## 应用创新赛道（目标：openKylin 上的 AI 应用/开发工具）

> 报名截止 **7/28 17:00**｜作品提交截止 **7/30 17:00**｜初赛评审 8 月上旬｜决赛 10 月北京｜承办：清华大学｜提交平台：GitLink
> ⚠️ 赛题名来自官方新闻页与 openKylin 社区页；**确切赛题描述/技术指标在登录态"附件2"PDF 中未获取**，凡未证实处均标注。**这是本批次截止最近的赛事（7/30）。**

---

## 1. 赛题要点（应用创新赛道说明 + 候选赛题短名单）

**赛道定位(官方要点)**："应用创新赛道主要聚焦操作系统之上的应用创新，包括与行业领域结合的应用开发、第三方应用的移植、行业解决方案、开发工具等"。来源：[cpipc 赛事新闻页](https://cpipc.acge.org.cn/cw/contestNews/detail/2c9080178c7c917b018d1b1a0af61cd6/2c9080179dcfa227019e297e87857f34?page=0)。

**本届应用创新赛道 4 个赛题（赛题名已确认，详细任务书/指标在附件2未获取）：**

| # | 赛题名 | 与 CV/AI 契合度 | 出处 |
|---|--------|---------------|------|
| 1 | 智能语录快报 | 中。内容聚合/生成，偏 NLP+多智能体 | [openKylin 社区页](https://www.openkylin.top/community/kylin_cup_13nd.html) |
| 2 | **智能视频剪辑软件** | **高**。可复用 YOLO 检测/实例分割做镜头识别、人物/物体跟踪、自动剪辑、字幕对齐；边缘加速可用 | 同上 |
| 3 | **个性化智能助手** | **高**。多智能体/多供应商工作流 + FastAPI 直接对口；可叠加端侧视觉 | 同上 |
| 4 | 基于 openKylin 和 RuyiBook 的 VSCode 移植 | 中。偏 RISC-V 编译移植，非 CV；有往届参考，工程确定性高 | 同上 + GitLink(见§3) |

> ⚠️ openKylin 社区页时间(2024-03~08-10)是**第十三届麒麟杯(2024)旧数据**，本届以 cpipc 官方(7/28/7/30)为准。赛题名两处一致、可信度高；但每题具体任务书、功能清单、硬性指标(架构/性能阈值/是否强制 RISC-V)**未获取，需登录 cpipc 下载附件2核实**。

**面向你背景的候选短名单（按推荐度）：**
- **首选：赛题3 个性化智能助手** —— 与多智能体/多供应商工作流+FastAPI 高度重合，可做 openKylin 桌面端"本地+云"混合助手，叠加端侧视觉(截图理解/文档图像识别)形成 CV 差异化。
- **次选：赛题2 智能视频剪辑** —— CV 技术栈最对口(检测/分割/跟踪)，论文潜力高，但工程量大、7/30 工期紧。
- **保底：赛题4 VSCode 移植** —— 有完整往届范本，确定性最高但 CV 含量低、难出论文，不推荐主攻。

## 2. 评分/评审标准

五个评审维度(来源：[cpipc 新闻页](https://cpipc.acge.org.cn/cw/contestNews/detail/2c9080178c7c917b018d1b1a0af61cd6/2c9080179dcfa227019e297e87857f34?page=0))：
1. **项目难度** 2. **项目完成度**(功能/正确性/性能/代码质量) 3. **项目创新性** 4. **开发过程**(GitLink 提交记录是评审依据——**不能临交前一次性 push**) 5. **技术报告**(作品链接/设计/实现/运行效果/演示视频/创新)。
各维度**具体分值权重未获取**(在附件2评分细则)。奖金：一等奖 3 支 10/7/4 万，决赛队各 3 万，二等各 2 万，三等各 6000 元。

## 3. 技术路线：openKylin 上做 AI 应用/开发工具

**openKylin 平台事实**：定位"深度集成 AI 的开源 OS"，麒麟 AI 助手支持云端/本地/私有化，软件商店可装本地 **Qwen2.5** 离线运行；基于 Debian/Ubuntu 系，**.deb 主流分发**(2.0 引入"开明"包格式)。来源：[麒麟软件新闻](https://kylinos.cn/about/news/1835.html)、[openKylin 官网](https://www.openkylin.top/)。

**主流技术栈**：
- 桌面 GUI：**Qt**(C++/PyQt/PySide)主流，官方有 [openKylin SDK 开发指南](https://docs.openkylin.top/zh/04_%E7%A4%BE%E5%8C%BA%E8%B4%A1%E7%8C%AE/%E5%BC%80%E5%8F%91%E6%8C%87%E5%8D%97/openKylin+SDK%E5%BC%80%E5%8F%91%E6%8C%87%E5%8D%97)；也可 Electron/Tauri。
- 打包/上架：`linuxdeployqt` 收依赖 + `dpkg-deb`/`dpkg-buildpackage` 打 .deb + control + .desktop。参考 [Kylin Qt 打包](https://blog.csdn.net/baidu_39064543/article/details/131477278)、[Qt 程序打 Deb](https://lichtg.github.io/post/022.html)。
- AI 推理后端：本地用 **Ollama / llama.cpp**(Qwen2.5/DeepSeek)；CV 用 **ONNX Runtime / OpenVINO**(x86 边缘)或 TensorRT(NVIDIA)；FastAPI 做本地推理服务(与你现有经验直通)。
- 多智能体编排：LangGraph / 自研多供应商路由(你已有)。

**可参考开源项目**：
- 赛题4 直接范本：[OSKYCX/jyokhrbdvcyz (GitLink)](https://www.gitlink.org.cn/OSKYCX/jyokhrbdvcyz)(往届 openKylin+RuyiBook RISC-V 上 VSCode 编译/移植/优化)。
- VSCode/RISC-V 移植细节：[Code-Server 在 RISC-V 移植实践](https://blog.gitcode.com/90f1adf3888e4f670c0e9aa68730718a.html)。
- 本地大模型：Ollama(github.com/ollama/ollama)、llama.cpp(github.com/ggml-org/llama.cpp)。CV：Ultralytics YOLO(你已熟)、ONNX Runtime。
- GitLink 往届题库入口：[OSKYCX 组织](https://www.gitlink.org.cn/OSKYCX/)。

> 注：上述推理库/框架为本简报基于公开做法的建议，非赛题强制要求。

## 4. 对照我的背景

**可直接复用**：多供应商/多智能体工作流+FastAPI → 直接做赛题3"个性化智能助手"后端与编排，几乎零学习成本；YOLO/实例分割/跟踪 → 赛题2"智能视频剪辑"镜头分析/自动打点；TensorRT+Jetson 经验 → 把作品定位"端侧/离线可运行"契合 openKylin 本地化；知识蒸馏 → 蒸馏小模型在 8G 边缘设备/openKylin 本地跑(创新+论文结合点)。RTX4070S 作训练机，Jetson/8G 边缘设备作演示端。

**需新做的增量**：① **openKylin 适配与打包**(优先 x86 桌面版，Qt/Electron GUI，`linuxdeployqt`+`dpkg-deb` 打 .deb)——最主要新增工作量，有官方文档；② **TensorRT 不通用**(仅 NVIDIA；纯 openKylin x86/边缘要改 **ONNX Runtime/OpenVINO**，Jetson 量化思路可迁移但后端要换)；③ **GitLink 协作流程**(持续 commit)；④(仅赛题4)RISC-V 交叉编译+RuyiBook，坑多。

> **🔧 硬件可行性（本轮新增 · 我无 RISC-V/RuyiBook 板卡）**
> 推荐的**赛题3 个性化智能助手 / 赛题2 智能视频剪辑**跑在 **openKylin x86** 上——你的 RTX4070S 主机**装 openKylin 虚拟机或双系统(x86)即可，无需额外硬件**；CV 推理走 ONNX Runtime/OpenVINO 或本机 GPU。
> **a) 仅用现有设备**：赛题3/2 完全可做、可演示、可提交，**不缺硬件**。
> **b) 唯一需硬件的是赛题4(openKylin+RuyiBook 在 RISC-V 上移植 VSCode)**：我无 RISC-V 板 → 需采购(LicheePi 4A / Milk-V 等 ≈ ¥500–1500，以电商为准)。**赛题4 CV 价值低 + 需采购 + RISC-V 交叉编译坑多 → 建议直接避开，主攻赛题3。**

## 5. 工作量粗估（按 3-4 人团队，约 6.5 周净工期到 7/30）

| 模块 | 赛题3 智能助手 | 赛题2 视频剪辑 |
|---|---|---|
| openKylin 环境/打包链路 | 1.0 | 1.0 |
| 核心 AI/CV 功能(复用为主) | 2.0 | 4.0 |
| GUI(Qt/Electron) | 2.0 | 3.0 |
| 端侧/本地推理优化(蒸馏/ONNX) | 1.5 | 2.0 |
| 测试+技术报告+演示视频 | 1.5 | 1.5 |
| **合计(人·周)** | **约 8** | **约 11.5** |

**换算**：赛题3 约 8 人·周 → **3 人 ≈ 2.7 周 / 4 人 ≈ 2 个日历周**可达可演示；赛题2 约 11.5 人·周 → 3 人 ≈ 4 周(风险高)。距 7/30 约 6.5 周 → **3-4 人做赛题3 时间充裕**，瓶颈是赛题细节未核实(附件2)+ GitLink 持续提交节奏，而非人力或硬件。

## 6. 推荐方案与主要风险

**推荐：主攻【赛题3 个性化智能助手】，以"端侧视觉能力 + 本地小模型(蒸馏)"作差异化创新。** 最大化复用你的多智能体/FastAPI 资产，工期可控；CV(截图/文档/图像理解)+蒸馏形成创新与论文增量；本地化契合 openKylin AI 定位。备选赛题2(论文潜力更高但工期险)。

**主要风险（尤其 7/30 极短工期）**：
1. **工期极短(7/28 报名、7/30 提交)**：本周内打通"openKylin 装机→Qt/Electron 跑通→.deb 打包"最小闭环，功能做减法。
2. **赛题细节未获取**：4 个赛题名已确认，但任务书/硬性指标在附件2——**必须尽快登录 cpipc 下载附件2核对**，否则可能做错方向。
3. **部署后端迁移**：TensorRT 不能直接搬到非 NVIDIA 的 openKylin/边缘机，预留时间改 ONNX Runtime/OpenVINO。
4. **"开发过程"评分**：GitLink 需从第一天起规律 commit，临交一次性 push 会丢分。
5. **资格/报名流程**：确认本人资格与 3 人+1 导师组队，及本校研究生院内部报名表流程(部分高校要求 6/30 前邮件提交，如 [中南大学通知](https://gra.csu.edu.cn/info/1038/40902.htm))。

## 7. 信息来源链接

- cpipc 赛事新闻页(赛道说明/评审/时间/奖金)：https://cpipc.acge.org.cn/cw/contestNews/detail/2c9080178c7c917b018d1b1a0af61cd6/2c9080179dcfa227019e297e87857f34?page=0
- openKylin 社区赛事页(4 赛题名)：https://www.openkylin.top/community/kylin_cup_13nd.html
- 中南大学组织通知(校内报名)：https://gra.csu.edu.cn/info/1038/40902.htm
- 湖南大学组织通知：https://gra.hnu.edu.cn/info/1050/10471.htm
- openKylin 官网：https://www.openkylin.top/
- 麒麟软件新闻(2.0/本地 Qwen2.5/开明包)：https://kylinos.cn/about/news/1835.html
- openKylin SDK 开发指南：https://docs.openkylin.top/zh/04_%E7%A4%BE%E5%8C%BA%E8%B4%A1%E7%8C%AE/%E5%BC%80%E5%8F%91%E6%8C%87%E5%8D%97/openKylin+SDK%E5%BC%80%E5%8F%91%E6%8C%87%E5%8D%97
- Qt 麒麟桌面应用流程：https://cloud.tencent.com/developer/article/2526365
- Kylin Qt 打包发布：https://blog.csdn.net/baidu_39064543/article/details/131477278
- Qt 程序打 Deb：https://lichtg.github.io/post/022.html
- GitLink 往届 VSCode/RuyiBook 项目：https://www.gitlink.org.cn/OSKYCX/jyokhrbdvcyz
- GitLink OSKYCX 组织(题库入口)：https://www.gitlink.org.cn/OSKYCX/
- Code-Server RISC-V 移植：https://blog.gitcode.com/90f1adf3888e4f670c0e9aa68730718a.html

---
**核心结论**：应用创新赛道 4 赛题名已确认、评审 5 维度与奖金已确认(均来自 cpipc 官方)。**但每题具体任务书/技术指标在登录态附件2 PDF 中未获取，务必尽快登录下载核对再定方向。** 推荐主攻"个性化智能助手"(复用多智能体+FastAPI)，CV 差异化用端侧视觉+蒸馏；最大风险是 7/30 极短工期与赛题细节未核实。
