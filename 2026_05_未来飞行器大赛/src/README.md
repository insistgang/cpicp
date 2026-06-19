# src · 报告书交付物生成器

> 05 未来飞行器大赛(交叉赛道 3.1)是**纯报告书赛**(不评 mAP、不需可运行系统)。
> 本目录的脚本把已有的 `项目报告书_初稿.md` 渲染成可提交的正式 Word，并生成报告所需配图。
> 全部基于已装包(matplotlib / python-docx / Pillow / numpy)，**离线可跑，无需 GPU**。

## 脚本

| 脚本 | 作用 | 产物 |
|------|------|------|
| `gen_figures.py` | matplotlib 生成 4 张报告配图(中文用 macOS 自带 CJK 字体) | `../output/figures/*.png` |
| `build_report_docx.py` | 解析初稿 Markdown，按官方模板骨架渲染正式 docx，封面/目录域/标题层级/原生表格/插图 | `../output/项目报告书_低空违建巡查系统.docx` |
| `build_attachment_docx.py` | 复用 `build_report_docx.py` 的通用解析+渲染管线，把附件1 TRL 报告渲染成 docx(附件封面/标题层级/原生表格/TRL 演进框图) | `../output/技术成熟度报告_附件1.docx` |
| `run_all_selftests.py` | 串起上面三个的自测 | — |

## 4 张配图

1. `fig_architecture.png` — 端-边-云三层系统总体架构框图(§2.1)
2. `fig_workflow.png` — 四阶段实施流程图，含闭环回流(§2.4)
3. `fig_business.png` — 商业模式三层金字塔 + 成本收益对比柱状(§4.2/§4.3)
4. `fig_crossmatrix.png` — 六学科 × 系统模块 交叉融合强度矩阵热力图(§5)

## 用法

```bash
# 生成全部配图
python3 gen_figures.py
# 渲染 Word 报告书(会自动插入上面 4 张图到对应章节)
python3 build_report_docx.py
# 渲染附件1 TRL 报告(复用同一解析+渲染管线; 也可 --input/--output 指定别的 md)
python3 build_attachment_docx.py
# 一键全自测(出图 + 正文渲染 + 附件渲染 + 读回校验)
python3 run_all_selftests.py
```

## 渲染要点

- **封面**：对齐官方模板字段(参赛队名称/单位/队员/联系人/日期)，待填项用蓝色提示文字，
  提交前按官方要求移除全部蓝色文字并匿名。
- **目录**：写入 Word `TOC` 域，打开文档右键“更新域”即生成真实目录。
- **页面**：A4，上下边距 2.54cm、左右 3.17cm；正文宋体小四、1.5 倍行距、首行缩进 2 字符；标题黑体。
- **表格**：Markdown 表格 → Word 原生表格(表头深蓝底白字)，初稿 7 张内容表 + 1 张封面信息表。
- **配图锚点**：按章节号 + 关键词匹配标题，在该节标题后插入对应 PNG(6 英寸宽) + 图题；
  若某图未命中锚点会在文末“附图汇总”兜底插入(当前 4 图均精确命中正文)。

## 提交前 PDF 转换(本机无 LibreOffice/Word headless 时)

docx 已生成。官方要求 PDF 提交：在 Word/WPS 打开 → 更新目录域 → 移除蓝色提示文字 →
另存为 PDF 即可。(本环境未装 office headless / 转换器，故 PDF 这一步留给有 Office 的机器完成。)
