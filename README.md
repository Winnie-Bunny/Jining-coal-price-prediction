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

GitHub Actions 会执行项目检查和接口测试，再构建 Docker 镜像并检查真实模型加载。Render 已连接到主分支，仅在 CI 检查通过后自动部署。API 访问密钥保存在 Render 服务的 Environment → API_KEY 中。免费试用与收费持久存储配置分开提供；添加配置文件不会创建云端资源。
