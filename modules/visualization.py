from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


# Windows 上一般有 Microsoft YaHei / SimHei；macOS 常见 Arial Unicode MS；Linux 环境下可能有 Noto Sans CJK。
# 这里不强制指定单一字体，避免不同电脑运行时报错。
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def _as_series(values) -> pd.Series:
    return pd.Series(values).reset_index(drop=True)


def plot_prediction(y_true, y_pred, max_points=300):
    y_true = _as_series(y_true)
    y_pred = _as_series(y_pred)
    n = min(len(y_true), int(max_points))

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(range(n), y_true.iloc[:n], label="真实功率")
    ax.plot(range(n), y_pred.iloc[:n], label="预测功率")
    ax.set_title("风力发电功率预测结果")
    ax.set_xlabel("测试样本序号")
    ax.set_ylabel("功率")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_error(y_true, y_pred, max_points=300):
    y_true = _as_series(y_true)
    y_pred = _as_series(y_pred)
    n = min(len(y_true), int(max_points))
    err = y_true.iloc[:n] - y_pred.iloc[:n]

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(range(n), err, label="预测误差")
    ax.axhline(0, linewidth=1)
    ax.set_title("预测误差曲线")
    ax.set_xlabel("测试样本序号")
    ax.set_ylabel("误差")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_feature_importance(importance_df, top_n=15):
    data = importance_df.head(int(top_n)).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.barh(data["feature"], data["importance"])
    ax.set_title("特征重要性排序")
    ax.set_xlabel("重要性")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return fig
