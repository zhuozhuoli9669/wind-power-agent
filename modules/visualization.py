from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import pandas as pd


def _setup_chinese_font() -> bool:
    """Set a Chinese-capable font for Matplotlib.

    Streamlit Cloud runs on Linux and often has no Chinese font by default.
    The repository includes packages.txt to install fonts-noto-cjk. This function
    also supports common Windows and macOS font paths for local use.
    """
    font_paths = [
        # Streamlit Cloud / Debian after installing fonts-noto-cjk
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        # Other common Linux Chinese fonts
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        # Windows
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        # macOS
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]

    for font_path in font_paths:
        path = Path(font_path)
        if path.exists():
            try:
                fm.fontManager.addfont(str(path))
                font_name = fm.FontProperties(fname=str(path)).get_name()
                plt.rcParams["font.family"] = "sans-serif"
                plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
                plt.rcParams["axes.unicode_minus"] = False
                return True
            except Exception:
                continue

    available_names = {f.name for f in fm.fontManager.ttflist}
    candidate_names = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "WenQuanYi Micro Hei",
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "PingFang SC",
        "Arial Unicode MS",
    ]
    for name in candidate_names:
        if name in available_names:
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return True

    plt.rcParams["axes.unicode_minus"] = False
    return False


CHINESE_FONT_AVAILABLE = _setup_chinese_font()


def _label(chinese: str, english: str) -> str:
    return chinese if CHINESE_FONT_AVAILABLE else english


def _as_series(values) -> pd.Series:
    return pd.Series(values).reset_index(drop=True)


def plot_prediction(y_true, y_pred, max_points=300):
    y_true = _as_series(y_true)
    y_pred = _as_series(y_pred)
    n = min(len(y_true), int(max_points))

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(range(n), y_true.iloc[:n], label=_label("真实功率", "True Power"))
    ax.plot(range(n), y_pred.iloc[:n], label=_label("预测功率", "Predicted Power"))
    ax.set_title(_label("风力发电功率预测结果", "Wind Power Prediction Result"))
    ax.set_xlabel(_label("测试样本序号", "Test Sample Index"))
    ax.set_ylabel(_label("功率", "Power"))
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
    ax.plot(range(n), err, label=_label("预测误差", "Prediction Error"))
    ax.axhline(0, linewidth=1)
    ax.set_title(_label("预测误差曲线", "Prediction Error Curve"))
    ax.set_xlabel(_label("测试样本序号", "Test Sample Index"))
    ax.set_ylabel(_label("误差", "Error"))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_feature_importance(importance_df, top_n=15):
    data = importance_df.head(int(top_n)).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.barh(data["feature"], data["importance"])
    ax.set_title(_label("特征重要性排序", "Feature Importance Ranking"))
    ax.set_xlabel(_label("重要性", "Importance"))
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return fig
