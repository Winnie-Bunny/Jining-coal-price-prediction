# 济宁气煤价格预测

[打开线上预测工作台](https://jining-coal-price.onrender.com)

LSTM 预测影响因素，SVR 预测 CCI 济宁气煤价格，支持 7 天、30 天和 365 天预测，并提供 SHAP 分析。

- [项目说明与运行步骤](project_materials_2026-10-16/README.md)
- [部署、自动测试与 Dashboard](deploy/README.md)
- [接口说明](project_materials_2026-10-16/docs/API.md)
- [数据与模型说明](project_materials_2026-10-16/docs/DATA.md)
- [实验报告](project_materials_2026-10-16/REPORT.md)
- [权利说明](project_materials_2026-10-16/NOTICE)

网页工作台和 API 同时启动：

```bash
python -m pip install -r inference/requirements.txt
python inference/app.py
```

打开 `http://127.0.0.1:7020`。页面支持上传 `.xlsx` / `.xls` / JSON，下载 Excel 模板，或仅将 Excel 转成 JSON。Python 3.11 推荐用于与 Docker 环境保持一致。

GitHub Actions 会执行项目检查和接口测试，再构建 Docker 镜像并检查真实模型加载。Render 已连接到主分支，仅在 CI 检查通过后自动部署。API 访问密钥保存在 Render 服务的 Environment → API_KEY 中，与 Render 平台的 `rnd_` API Key 不同。免费试用与收费持久存储配置分开提供；添加配置文件不会创建云端资源。

## 上传 Excel 预测

1. 打开线上工作台，下载「Excel 填写模板」。也可从仓库[下载模板](inference/static/downloads/coal-price-template.xlsx)。
2. 在「数据」工作表填写历史数据，保留第一行列名。
3. 输入服务访问密钥，选择预测周期，上传 `.xlsx` 或 `.xls`。
4. 点击「开始预测」，等待完成后查看价格曲线、逐日价格和主要因素，也可下载结果 JSON。

如只需转换数据，点击「转换并下载 JSON」，不会创建预测任务。原有 JSON 上传方式继续可用。

| 预测周期 | 至少需要的连续历史数据 |
| --- | --- |
| 未来 7 天 | 60 天 |
| 未来 30 天 | 90 天 |
| 未来 365 天 | 364 天 |

Excel 必须包含 **日期、历史价格及全部 211 项因素**（共 213 列）。支持模板中文列名或 API 英文代码，列顺序可调整。系统按日期排序并选择最近所需天数；完全相同的重复日期记录会合并，冲突记录会报错。日期须按天连续，数值须完整且有效。只有煤价一列的 Excel 不能直接运行现有模型。

优先读取名为「数据」的工作表，否则读取第一张表。单次上传上限 8 MB，最多 5000 行数据、512 列。公式需先在 Excel 中计算并保存，或粘贴为数值。字段和错误提示见[接口文档](project_materials_2026-10-16/docs/API.md)。

页面仅在当前内存中保留访问密钥。原始 Excel 在内存中处理；任务和预测结果保存在服务的状态目录中。

## 项目结构

| 路径 | 作用 |
| --- | --- |
| `inference/app.py` | Flask API、任务管理和网页入口 |
| `inference/excel_import.py` | Excel 读取、字段校验及 JSON 转换 |
| `inference/static/` | 中文预测工作台及 Excel 模板 |
| `inference/{week2week,month2month,year2year}/` | 三种周期的预测模型与预处理 |
| `inference/forecast_explanations.py` | 控制 SHAP 计算量及内存占用 |
| `data/`、`model/`、`lstm_models/` | 历史数据、模型及缩放器 |
| `predict_week.py`、`predict_month.py`、`predict_year.py`、`svr_price.py` | 训练和实验脚本 |
| `project_materials_2026-10-16/` | 详细说明、接口文档、样例和测试 |
| `deploy/` | Render 配置及部署说明 |
| `.github/workflows/ci-cd.yml` | 自动测试、Docker 构建及镜像发布 |

## Docker 运行

在项目根目录执行：

```bash
docker build -f inference/dockerfile -t cci-jining ./inference
docker run --rm -p 127.0.0.1:7020:7020 cci-jining
```

## API

- `GET /health`：检查模型是否就绪。
- `POST /convert`：上传 Excel，返回校验后的预测请求 JSON 和数据摘要。
- `POST /tasks`：提交 JSON 或 Excel，创建异步预测任务。
- `GET /tasks/<task_id>`：查询任务状态和预测结果。

配置 `API_KEY` 后，转换和任务接口需发送 `Authorization: Bearer <服务访问密钥>`。Excel 使用 multipart 字段 `id`（7、30 或 365）和 `file`。完整示例见[接口文档](project_materials_2026-10-16/docs/API.md)。

## 验证与运行限制

Excel 功能已通过 17 项自动测试；GitHub Actions 在 **512 MB 内存限制**下验证了真实 Excel 转换、完整七日预测及 SHAP 分析。线上也已验证转换结果与原始 JSON 输入一致，直接上传 Excel 可完成真实七日预测。

完整月度/年度预测、预测准确率和独立回测尚未验证；模型可运行不等于预测准确。示例数据为截至 2025 年的历史数据，不是实时行情。

当前免费 Render 服务没有持久磁盘，重启或重新部署可能丢失任务记录和结果；同一时间处理一个预测任务。详细配置见[部署说明](deploy/README.md)。
