#!/usr/bin/env python3
"""
crossdomain_eval.py · "陆→海"跨域迁移评价【实验设计模板 + 接口/占位】

为什么是你的差异化优势:
  你的学位论文做"无人机违建检测的跨域迁移评价(跨域迁移评价方法)"。本赛题天然是一个跨域问题:
    source(陆域，违建/VisDrone person 等大量标注) → target(海域水面救援，标注少)。
  把你那套"域差度量 + 跨域泛化评价"迁移过来，既提分(指导增广/微调)，又是论文创新点。

本文件提供三块的接口与占位实现(TODO)，结构完整、可 import 运行(用随机特征演示)，
你按论文方法替换 TODO 即可:
  1. 特征提取: 用检测主干 backbone 提取 source/target 图像特征
  2. 域差度量: MMD / CORAL / A-distance / 特征分布可视化(t-SNE)
  3. 跨域泛化评价: source-only 模型在 target 上的掉点 + 你的评价指标

用法(演示):
  python crossdomain_eval.py --demo
  python crossdomain_eval.py --source-weights land.pt --target-data configs/searescue.yaml
"""
import argparse
import numpy as np


# ---------- 2. 域差度量 ----------
def mmd_rbf(X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> float:
    """最大均值差异(MMD, RBF 核)。X:source 特征 [n,d], Y:target 特征 [m,d]。值越大域差越大。"""
    def rbf(A, B):
        # ||a-b||^2
        aa = (A**2).sum(1)[:, None]
        bb = (B**2).sum(1)[None, :]
        d2 = aa + bb - 2 * A @ B.T
        return np.exp(-gamma * d2)
    return float(rbf(X, X).mean() + rbf(Y, Y).mean() - 2 * rbf(X, Y).mean())


def coral_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """CORAL: source/target 二阶统计(协方差)差异的 Frobenius 范数。"""
    cs = np.cov(X, rowvar=False)
    ct = np.cov(Y, rowvar=False)
    return float(np.linalg.norm(cs - ct, ord="fro") ** 2 / (4 * X.shape[1] ** 2))


# ---------- 1. 特征提取(接口，TODO 接你的 backbone) ----------
def extract_features(weights: str, data_or_dir: str, max_imgs: int = 200) -> np.ndarray:
    """TODO[实现]: 加载检测模型 backbone，对图像取全局池化特征 [N, d]。
    建议: ultralytics YOLO 取 model.model[:10] 的输出做 GAP；或单独 load 一个 backbone。
    现返回占位随机特征以保证流程可跑。"""
    print(f"  [TODO] 用 {weights} 对 {data_or_dir} 提特征(占位随机)")
    rng = np.random.default_rng(0)
    return rng.normal(size=(max_imgs, 256)).astype("float32")


# ---------- 3. 跨域泛化评价 ----------
def cross_domain_report(source_weights, target_data, demo=False):
    if demo:
        rng = np.random.default_rng(1)
        Xs = rng.normal(0.0, 1.0, (200, 256)).astype("float32")     # source
        Xt = rng.normal(0.6, 1.2, (200, 256)).astype("float32")     # target(有偏移)
    else:
        Xs = extract_features(source_weights, "raw/source_land")     # 陆域
        Xt = extract_features(source_weights, target_data)           # 海域(用同一模型提，看分布漂移)

    mmd = mmd_rbf(Xs, Xt, gamma=1.0 / Xs.shape[1])
    coral = coral_distance(Xs, Xt)
    print("\n=== 陆→海 域差度量 ===")
    print(f"  MMD(RBF)      = {mmd:.4f}   (越大域差越大)")
    print(f"  CORAL 距离    = {coral:.4f}")
    print("\n=== 实验设计模板(论文用) ===")
    print("  E1 source-only: 仅陆域训 → 海域测，记录 mAP 掉点(域差上界证据)")
    print("  E2 target-ft  : 陆域预训练 + 海域少量微调 → 掉点恢复多少")
    print("  E3 域适应     : 加 MMD/CORAL 对齐或对抗域适应 → 对比 E1/E2")
    print("  E4 你的评价指标: 用学位论文的跨域迁移评价方法量化'迁移难度/可迁移性'，"
          "与 MMD/CORAL 相关性分析 → 创新点")
    print("  TODO[实现]: extract_features 接真实 backbone; 补 t-SNE 可视化; 接 E1-E4 训练脚本")
    # TODO[登录核对]: 若赛题提供官方海域数据，把 target 换成官方集，域差结论更可信。


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-weights", default="")
    ap.add_argument("--target-data", default="configs/searescue.yaml")
    ap.add_argument("--demo", action="store_true", help="用随机特征演示流程")
    a = ap.parse_args()
    cross_domain_report(a.source_weights, a.target_data, demo=a.demo or not a.source_weights)


if __name__ == "__main__":
    main()
