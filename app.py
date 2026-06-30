from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from modules.data_process import (
    build_supervised_dataset,
    clean_dataframe,
    infer_columns,
    read_csv_safely,
)
from modules.model_train import feature_importance, train_and_evaluate
from modules.report_agent import DEFAULT_ENDPOINT, call_qwen_report_with_status
from modules.visualization import plot_error, plot_feature_importance, plot_prediction


st.set_page_config(
    page_title="WindPower Agent",
    page_icon="🌬️",
    layout="wide",
)

st.title("WindPower Agent：风力发电功率预测与智能分析系统")
st.caption("基于 LLM + 工具 + 工作流的风电数据自动建模、评估、可视化与报告生成智能体。")

AGENT_STEPS = [
    "任务理解：识别用户需求为风力发电功率预测与数据科学分析任务",
    "数据接入：读取 CSV 数据并完成字段识别",
    "数据治理：处理时间列、缺失值、重复记录和异常功率值",
    "特征工程：构建时间周期特征与历史功率滞后特征",
    "模型训练：调用多个机器学习回归模型进行对比实验",
    "指标评估：计算 MAE、RMSE、MAPE 和 R² 并选择最优模型",
    "可视化分析：生成预测曲线、误差曲线和特征重要性图",
    "智能报告：调用阿里云 Qwen 或本地模板生成专业分析报告",
]

with st.sidebar:
    st.header("参数设置")
    horizon = st.number_input("预测步长", min_value=1, max_value=24, value=1, step=1)
    test_ratio = st.slider("测试集比例", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    max_points = st.slider("曲线展示点数", min_value=100, max_value=1000, value=300, step=50)

    st.divider()
    st.subheader("阿里云 Qwen 设置")
    api_key = st.text_input("DASHSCOPE_API_KEY", type="password", value=os.getenv("DASHSCOPE_API_KEY", ""))
    model_name = st.text_input("模型名称", value=os.getenv("DASHSCOPE_MODEL", "qwen-plus"))
    endpoint = st.text_input("接口地址", value=os.getenv("DASHSCOPE_ENDPOINT", DEFAULT_ENDPOINT))
    st.caption("不填写 API Key 时，系统会使用本地模板生成报告；填写后将调用阿里云 Qwen。")

st.info("本系统的智能体逻辑：由大语言模型承担报告生成与解释层，由数据处理、建模、评估和可视化工具承担可执行计算层。")

with st.expander("查看智能体工作流设计", expanded=False):
    for i, step in enumerate(AGENT_STEPS, start=1):
        st.write(f"{i}. {step}")

uploaded_file = st.file_uploader("上传风电历史数据 CSV", type=["csv"])
use_sample = st.checkbox("使用示例数据", value=uploaded_file is None)

if uploaded_file is None and not use_sample:
    st.info("请上传 CSV 文件，或勾选使用示例数据。")
    st.stop()

try:
    if use_sample and uploaded_file is None:
        df = pd.read_csv(Path(__file__).parent / "sample_data.csv")
        source_name = "sample_data.csv"
    else:
        df = read_csv_safely(uploaded_file)
        source_name = uploaded_file.name
except Exception as exc:
    st.error(f"读取 CSV 失败：{exc}")
    st.stop()

st.subheader("1. 数据预览")
col_a, col_b, col_c = st.columns(3)
col_a.metric("数据来源", source_name)
col_b.metric("样本数", len(df))
col_c.metric("字段数", len(df.columns))
st.dataframe(df.head(20), use_container_width=True)

try:
    info = infer_columns(df)
except Exception as exc:
    st.error(f"字段识别失败：{exc}")
    st.stop()

st.subheader("2. 字段识别结果")
col1, col2, col3 = st.columns(3)
col1.write("**自动识别时间列**")
col1.code(info.time_col if info.time_col else "未识别")
col2.write("**自动识别目标功率列**")
col2.code(info.target_col)
col3.write("**可用数值特征数**")
col3.code(str(len(info.feature_cols)))

with st.expander("手动调整字段", expanded=False):
    columns = list(df.columns)
    time_options = ["无"] + columns
    default_time_index = time_options.index(info.time_col) if info.time_col in time_options else 0
    selected_time = st.selectbox("时间列", time_options, index=default_time_index)
    selected_target = st.selectbox("目标功率列", columns, index=columns.index(info.target_col))
    st.caption("选择后会直接用于下方的数据处理和模型训练。")

info.time_col = None if selected_time == "无" else selected_time
info.target_col = selected_target
info.feature_cols = [c for c in info.numeric_cols if c != info.target_col]

try:
    cleaned, logs = clean_dataframe(df, info.time_col, info.target_col)
    X, y, supervised, feature_cols = build_supervised_dataset(
        cleaned,
        target_col=info.target_col,
        feature_cols=info.feature_cols,
        horizon=int(horizon),
        lags=[1, 2, 3, 4, 6, 12],
    )
except Exception as exc:
    st.error(f"数据处理失败：{exc}")
    st.stop()

st.subheader("3. 数据质量检测与特征构建")
missing_table = df.isna().sum().reset_index()
missing_table.columns = ["字段", "缺失值数量"]

col_left, col_right = st.columns(2)
with col_left:
    st.write("**缺失值统计**")
    st.dataframe(missing_table, use_container_width=True)
with col_right:
    st.write("**处理日志**")
    if logs:
        for item in logs:
            st.write("- " + item)
    else:
        st.write("未发现需要特殊处理的问题。")
    st.write(f"监督学习样本数：**{len(X)}**")
    st.write(f"最终特征数：**{len(feature_cols)}**")

st.subheader("4. 智能体工作流执行")
if st.button("开始训练并生成分析", type="primary"):
    workflow_box = st.container(border=True)
    with workflow_box:
        st.write("**执行日志**")
        for step in AGENT_STEPS[:4]:
            st.write(f"✅ {step}")
        st.write("⏳ 模型训练：正在调用机器学习模型训练工具……")

    with st.spinner("模型训练与评估中..."):
        result = train_and_evaluate(X, y, test_ratio=float(test_ratio))

    with workflow_box:
        st.write(f"✅ 模型训练：已完成 LinearRegression、Ridge、DecisionTree、RandomForest 的训练")
        st.write("✅ 指标评估：已计算 MAE、RMSE、MAPE 和 R²")
        st.write("✅ 可视化分析：已生成预测曲线、误差曲线和特征重要性图")

    st.subheader("5. 模型对比与最优模型选择")
    metrics_df = result["metrics_df"].copy()
    st.dataframe(
        metrics_df.style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "MAPE(%)": "{:.2f}", "R2": "{:.4f}"}),
        use_container_width=True,
    )

    best_row = metrics_df.iloc[0]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("最优模型", result["best_model_name"])
    m2.metric("MAE", f"{best_row['MAE']:.4f}")
    m3.metric("RMSE", f"{best_row['RMSE']:.4f}")
    m4.metric("MAPE(%)", f"{best_row['MAPE(%)']:.2f}" if pd.notna(best_row["MAPE(%)"]) else "N/A")
    m5.metric("R²", f"{best_row['R2']:.4f}")

    st.success(f"当前最优模型：{result['best_model_name']}。系统按照 RMSE 最小原则完成自动选择。")

    st.subheader("6. 预测结果可视化")
    fig1 = plot_prediction(result["y_test"], result["best_pred"], max_points=max_points)
    fig2 = plot_error(result["y_test"], result["best_pred"], max_points=max_points)
    st.pyplot(fig1)
    st.pyplot(fig2)

    st.subheader("7. 特征重要性")
    imp = feature_importance(result["best_model"], feature_cols)
    st.dataframe(imp.head(20), use_container_width=True)
    fig3 = plot_feature_importance(imp, top_n=min(15, len(imp)))
    st.pyplot(fig3)

    pred_df = pd.DataFrame({
        "y_true": pd.Series(result["y_test"]).reset_index(drop=True),
        "y_pred": pd.Series(result["best_pred"]).reset_index(drop=True),
        "error": pd.Series(result["y_test"]).reset_index(drop=True) - pd.Series(result["best_pred"]).reset_index(drop=True),
    })

    st.subheader("8. 智能分析报告")
    data_summary = {
        "rows": len(df),
        "columns": len(df.columns),
        "target_col": info.target_col,
        "time_col": info.time_col,
        "horizon": int(horizon),
        "samples_after_processing": len(X),
        "feature_count": len(feature_cols),
    }
    feature_summary = ", ".join(imp.head(10)["feature"].astype(str).tolist())
    report, report_source = call_qwen_report_with_status(
        metrics_df=metrics_df,
        best_model=result["best_model_name"],
        data_summary=data_summary,
        feature_summary=feature_summary,
        api_key=api_key,
        model_name=model_name,
        endpoint=endpoint,
    )
    with workflow_box:
        st.write(f"✅ 智能报告：{report_source}")
        st.write("✅ 工作流执行完成")

    st.caption(f"报告来源：{report_source}")
    st.markdown(report)

    st.subheader("9. 结果下载")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button(
            "下载预测结果 CSV",
            data=pred_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="prediction_result.csv",
            mime="text/csv",
        )
    with d2:
        st.download_button(
            "下载模型评估 CSV",
            data=metrics_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="model_metrics.csv",
            mime="text/csv",
        )
    with d3:
        st.download_button(
            "下载分析报告 Markdown",
            data=report.encode("utf-8"),
            file_name="wind_power_analysis_report.md",
            mime="text/markdown",
        )
else:
    st.info("确认字段和参数后，点击“开始训练并生成分析”。")
