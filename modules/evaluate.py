import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    # 功率数据常出现接近 0 的时段，直接计算 MAPE 会被极小分母放大。
    # 这里仅在真实功率大于额定量级 5% 的样本上计算 MAPE。
    scale = max(float(np.nanmax(np.abs(y_true))), 1e-6)
    mask = np.abs(y_true) > 0.05 * scale
    if mask.sum() >= 5:
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / np.abs(y_true[mask]))) * 100)
    else:
        mape = float("nan")

    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE(%)": mape,
        "R2": float(r2),
    }
