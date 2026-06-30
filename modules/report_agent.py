from __future__ import annotations

import os
from typing import Dict, Tuple

import requests


DEFAULT_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"


def build_report_prompt(metrics_table: str, best_model: str, data_summary: Dict, feature_summary: str) -> str:
    return f"""
你是一名新能源数据科学工程师。请根据以下风力发电功率预测实验结果，生成一段课程作业可用的专业分析报告。
要求：
1. 用中文；
2. 分为“数据情况、模型表现、误差分析、应用价值、改进方向”五部分；
3. 不要编造未给出的具体数值；
4. 语言适合研究生课程论文或系统展示；
5. 强调该系统是“LLM + 工具 + 工作流”的智能体应用，不是单纯的预测脚本。

【数据概况】
{data_summary}

【模型评估表】
{metrics_table}

【最优模型】
{best_model}

【主要特征】
{feature_summary}
""".strip()


def fallback_report(metrics_df, best_model: str, data_summary: Dict, feature_summary: str) -> str:
    best = metrics_df.iloc[0].to_dict()
    return f"""
### 数据情况
本系统对上传的风力发电历史数据进行了字段识别、缺失值处理、时间排序、滞后特征构建和监督学习样本转换。原始数据规模为 {data_summary.get('rows')} 行、{data_summary.get('columns')} 个字段，目标变量为 `{data_summary.get('target_col')}`，预测步长为 {data_summary.get('horizon')}。处理后可用于建模的监督学习样本数为 {data_summary.get('samples_after_processing')}。

### 模型表现
系统对多个回归模型进行了训练和对比，当前最优模型为 **{best_model}**。该模型在测试集上的 RMSE 为 **{best.get('RMSE'):.4f}**，MAE 为 **{best.get('MAE'):.4f}**，R² 为 **{best.get('R2'):.4f}**。从指标结果看，该模型能够较好地学习风电功率与气象变量、历史功率特征之间的非线性关系。

### 误差分析
风力发电功率受风速、风向、气象扰动和机组运行状态影响，具有明显的随机性和波动性。预测误差通常集中在风速快速变化、机组出力突变或原始数据存在噪声的时段。若后续加入更长的历史窗口、天气预报变量以及异常工况识别模块，模型稳定性仍有提升空间。

### 应用价值
该系统采用“LLM + 工具 + 工作流”的智能体思路，将数据清洗、特征构建、模型训练、指标评估、可视化展示和自然语言解释整合为一个自动化流程。用户不需要手动编写完整的数据科学代码，即可完成从原始数据到预测结论的分析过程，降低了风电功率预测建模的使用门槛。

### 改进方向
后续可以进一步接入 LSTM、Transformer 或 TFT 等深度时序模型，同时加入领域知识库、异常检测规则和模型自动选择机制，使智能体能够根据不同风电场数据自动调整建模策略。

**主要参考特征：** {feature_summary}
""".strip()


def call_qwen_report_with_status(
    metrics_df,
    best_model: str,
    data_summary: Dict,
    feature_summary: str,
    api_key: str = "",
    model_name: str = "qwen-plus",
    endpoint: str = DEFAULT_ENDPOINT,
) -> Tuple[str, str]:
    """返回: (报告正文, 报告来源说明)。"""
    if not api_key:
        api_key = os.getenv("DASHSCOPE_API_KEY", "")
    if not api_key:
        return fallback_report(metrics_df, best_model, data_summary, feature_summary), "本地模板报告（未填写 DASHSCOPE_API_KEY）"

    prompt = build_report_prompt(
        metrics_table=metrics_df.to_markdown(index=False),
        best_model=best_model,
        data_summary=data_summary,
        feature_summary=feature_summary,
    )

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "你是一个面向能源预测任务的数据科学智能体。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(endpoint, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"], f"阿里云 Qwen 生成报告（模型：{model_name}）"
    except Exception as exc:
        report = fallback_report(metrics_df, best_model, data_summary, feature_summary)
        report += f"\n\n> 大模型接口暂未成功调用，系统已自动切换为本地报告模板。接口错误：{exc}"
        return report, "本地模板报告（Qwen 接口调用失败）"


def call_qwen_report(*args, **kwargs) -> str:
    report, _ = call_qwen_report_with_status(*args, **kwargs)
    return report
