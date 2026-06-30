from typing import Dict

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from .evaluate import regression_metrics


def chronological_split(X: pd.DataFrame, y: pd.Series, test_ratio: float = 0.2):
    n = len(X)
    split = max(1, min(n - 1, int(n * (1 - test_ratio))))
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


def get_models(random_state: int = 42) -> Dict[str, object]:
    return {
        "LinearRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),
        "DecisionTree": DecisionTreeRegressor(
            max_depth=10,
            min_samples_leaf=5,
            random_state=random_state,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=80,
            max_depth=10,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=1,
        ),
    }


def train_and_evaluate(X: pd.DataFrame, y: pd.Series, test_ratio: float = 0.2):
    X_train, X_test, y_train, y_test = chronological_split(X, y, test_ratio)
    models = get_models()
    results = []
    fitted = {}
    predictions = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics = regression_metrics(y_test, pred)
        row = {"Model": name, **metrics}
        results.append(row)
        fitted[name] = model
        predictions[name] = pred

    metrics_df = pd.DataFrame(results).sort_values("RMSE", ascending=True).reset_index(drop=True)
    best_name = metrics_df.iloc[0]["Model"]

    return {
        "metrics_df": metrics_df,
        "best_model_name": best_name,
        "best_model": fitted[best_name],
        "y_test": y_test,
        "best_pred": predictions[best_name],
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
    }


def feature_importance(model, feature_names):
    real_model = model
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        real_model = model.named_steps["model"]

    if hasattr(real_model, "feature_importances_"):
        return pd.DataFrame({
            "feature": feature_names,
            "importance": real_model.feature_importances_,
        }).sort_values("importance", ascending=False)

    if hasattr(real_model, "coef_"):
        return pd.DataFrame({
            "feature": feature_names,
            "importance": abs(real_model.coef_),
        }).sort_values("importance", ascending=False)

    return pd.DataFrame({"feature": feature_names, "importance": [0.0] * len(feature_names)})
