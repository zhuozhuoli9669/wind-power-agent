import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ColumnInfo:
    time_col: Optional[str]
    target_col: str
    feature_cols: List[str]
    numeric_cols: List[str]


def read_csv_safely(file) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "gbk", "gb2312", "latin1"]
    last_error = None
    for enc in encodings:
        try:
            return pd.read_csv(file, encoding=enc)
        except Exception as exc:
            last_error = exc
            try:
                file.seek(0)
            except Exception:
                pass
    raise last_error


def _score_time_column(series: pd.Series) -> float:
    parsed = pd.to_datetime(series, errors="coerce")
    return float(parsed.notna().mean())


def infer_columns(df: pd.DataFrame) -> ColumnInfo:
    if df.empty:
        raise ValueError("数据为空，无法分析。")

    columns = list(df.columns)
    lower_map = {c: str(c).lower() for c in columns}

    time_candidates = []
    for c in columns:
        name = lower_map[c]
        if any(k in name for k in ["time", "date", "timestamp", "datetime", "时间", "日期"]):
            time_candidates.append(c)

    time_col = None
    best_score = 0.0
    for c in time_candidates or columns:
        score = _score_time_column(df[c])
        if score > best_score:
            best_score = score
            time_col = c
    if best_score < 0.75:
        time_col = None

    numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        for c in columns:
            converted = pd.to_numeric(df[c], errors="coerce")
            if converted.notna().mean() > 0.7:
                numeric_cols.append(c)

    target_keywords = ["power", "active_power", "generation", "output", "kw", "功率", "发电", "出力"]
    target_candidates = []
    for c in numeric_cols:
        name = lower_map[c]
        if any(k in name for k in target_keywords):
            target_candidates.append(c)

    if target_candidates:
        # 优先选择数据波动较明显的功率列
        target_col = max(target_candidates, key=lambda x: pd.to_numeric(df[x], errors="coerce").std(skipna=True))
    else:
        target_col = max(numeric_cols, key=lambda x: pd.to_numeric(df[x], errors="coerce").std(skipna=True))

    feature_cols = [c for c in numeric_cols if c != target_col]
    return ColumnInfo(time_col=time_col, target_col=target_col, feature_cols=feature_cols, numeric_cols=numeric_cols)


def clean_dataframe(df: pd.DataFrame, time_col: Optional[str], target_col: str) -> Tuple[pd.DataFrame, List[str]]:
    data = df.copy()
    logs = []

    if time_col:
        data[time_col] = pd.to_datetime(data[time_col], errors="coerce")
        before = len(data)
        data = data.dropna(subset=[time_col])
        dropped = before - len(data)
        if dropped:
            logs.append(f"删除无法解析时间的记录 {dropped} 条。")
        data = data.sort_values(time_col).drop_duplicates(subset=[time_col], keep="first")
        data = data.set_index(time_col)
    else:
        data.index = pd.RangeIndex(len(data))
        logs.append("未识别到可靠时间列，已按原始行顺序建模。")

    for c in data.columns:
        if not pd.api.types.is_numeric_dtype(data[c]):
            converted = pd.to_numeric(data[c], errors="coerce")
            if converted.notna().mean() > 0.7:
                data[c] = converted

    numeric_cols = [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c])]
    for c in numeric_cols:
        missing_before = int(data[c].isna().sum())
        if missing_before:
            data[c] = data[c].interpolate(limit_direction="both")
            data[c] = data[c].fillna(data[c].median())

    if target_col in data.columns:
        neg_count = int((pd.to_numeric(data[target_col], errors="coerce") < 0).sum())
        if neg_count:
            data[target_col] = data[target_col].clip(lower=0)
            logs.append(f"目标功率列存在负值 {neg_count} 条，已截断为 0。")

    return data, logs


def add_time_features(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        hour = out.index.hour + out.index.minute / 60.0
        month = out.index.month
        dow = out.index.dayofweek
        out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        out["month_sin"] = np.sin(2 * np.pi * month / 12)
        out["month_cos"] = np.cos(2 * np.pi * month / 12)
        out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    return out


def add_lag_features(data: pd.DataFrame, target_col: str, lags: List[int]) -> pd.DataFrame:
    out = data.copy()
    for lag in lags:
        out[f"{target_col}_lag_{lag}"] = out[target_col].shift(lag)
    out[f"{target_col}_roll_mean_4"] = out[target_col].rolling(4).mean()
    out[f"{target_col}_roll_std_4"] = out[target_col].rolling(4).std()
    return out


def build_supervised_dataset(
    data: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    horizon: int = 1,
    lags: Optional[List[int]] = None,
):
    if lags is None:
        lags = [1, 2, 3, 4]

    work = data.copy()
    base_features = [c for c in feature_cols if c in work.columns and c != target_col]
    work = add_time_features(work)
    work = add_lag_features(work, target_col, lags)

    time_features = [c for c in work.columns if re.match(r"^(hour|month|dow)_", c)]
    lag_features = [c for c in work.columns if c.startswith(f"{target_col}_lag_") or c.startswith(f"{target_col}_roll_")]
    final_features = []
    for c in base_features + time_features + lag_features:
        if c not in final_features and pd.api.types.is_numeric_dtype(work[c]):
            final_features.append(c)

    work["target_future"] = work[target_col].shift(-horizon)
    work = work.dropna(subset=final_features + ["target_future"])

    if len(work) < 30:
        raise ValueError("有效样本太少，无法完成训练。请检查数据量、目标列或预测步长设置。")

    X = work[final_features]
    y = work["target_future"]
    return X, y, work, final_features
