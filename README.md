# WindPower Agent：风力发电功率预测与智能分析系统

## 1. 项目简介

本项目是一个面向风力发电功率预测任务的专业智能体应用。系统支持上传风电历史数据 CSV，自动完成数据读取、字段识别、数据清洗、特征构建、模型训练、指标评估、可视化展示，并可调用阿里云 Qwen 大模型生成专业分析报告。

项目定位不是普通的预测脚本，而是“LLM + 工具 + 工作流”的智能体系统：

- LLM 层：承担结果解释、报告生成和专业语言组织；
- 工具层：承担数据清洗、特征工程、模型训练、指标评估和可视化；
- 工作流层：将专业数据科学流程串联为可交互的 Web 应用。

## 2. 主要功能

- CSV 数据上传与示例数据测试
- 时间列、目标功率列自动识别，并支持手动调整
- 缺失值、负功率、重复时间戳等数据质量处理
- 时间周期特征与历史功率滞后特征构建
- 多模型训练与对比：LinearRegression、Ridge、DecisionTree、RandomForest
- MAE、RMSE、MAPE、R² 指标评估
- 按 RMSE 自动选择最优模型
- 中文预测曲线、误差曲线和特征重要性图
- 智能体工作流执行日志
- 阿里云 Qwen 或本地模板生成分析报告
- 下载预测结果、模型评估结果和分析报告

## 3. 本地运行

进入项目目录后执行：

```bash
pip install -r requirements.txt
streamlit run app.py
```

浏览器打开后，勾选“使用示例数据”，然后点击“开始训练并生成分析”。

## 4. 阿里云 Qwen 配置

可以在系统侧边栏输入 API Key，也可以在终端设置环境变量。

Windows：

```bash
set DASHSCOPE_API_KEY=你的APIKey
set DASHSCOPE_MODEL=qwen-plus
streamlit run app.py
```

macOS 或 Linux：

```bash
export DASHSCOPE_API_KEY=你的APIKey
export DASHSCOPE_MODEL=qwen-plus
streamlit run app.py
```

不配置 API Key 时，系统仍可运行，并使用本地模板生成报告。

## 5. 数据格式建议

CSV 中建议包含以下字段中的若干项：

- 时间：time / datetime / timestamp / 时间
- 功率：power / active_power / generation / 功率
- 风速：wind_speed
- 风向：wind_direction
- 温度：temperature
- 湿度：humidity
- 气压：pressure

系统会自动识别功率目标列，也支持在页面中手动调整。

## 6. 课程论文可写的技术路线

用户上传数据 → 字段识别 → 数据质量检测 → 特征工程 → 多模型训练 → 指标评估 → 最优模型选择 → 可视化展示 → LLM 生成分析报告。

该流程可以在论文中概括为：面向风力发电功率预测任务的智能体工作流设计。
