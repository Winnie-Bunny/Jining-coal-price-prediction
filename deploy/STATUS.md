# 部署状态 — 2026-10-16

- 线上网页与 API：https://jining-coal-price.onrender.com
- Render 服务：https://dashboard.render.com/web/srv-db99m6jrjlhs73e01j40
- 工作区：My Workspace；免费方案，512 MB 内存，没有持久磁盘。未创建收费资源。
- PR #1 与 #2 已通过检查并合并；当前已部署代码 df1b653c2802e3c7ee51b2a6655100d2961e35ad。
- 本地 9 项接口、数据和因素分析测试通过。
- GitHub 完整性检查、Docker 构建、三个周期的模型加载与前向计算通过。
- GitHub 在 512 MB 容器限制下完成了真实 7 天预测与 SHAP 分析。
- Render 修复后也完成真实 7 天预测；任务 2016f0b1-2508-4243-a4bc-7df912fca243。网页、健康检查和未授权请求拦截已验证。
- 月、年模型已验证加载和前向计算，尚未验证完整的月、年预测请求；预测准确性未因此得到验证。

## 自动部署已配置

Render 自动部署策略为 After CI Checks Pass，主分支检查全部通过后才上线。健康检查路径为 /health。该原生流程不需要 GitHub 保存 Render 部署钩子密钥；已移除未使用的可选钩子步骤，避免重复部署。

API_KEY 已配置在 Render 的 Environment 中。请在 Render 页面读取后填入网页工作台；不要写进仓库或公开聊天。

默认因素分析使用最多 16 条背景记录、256 个掩码样本，是低内存 SHAP 估算。最初的默认 SHAP 配置曾在免费实例上触发内存超限；PR #2 已修复并实测通过。煤价计算逻辑未改动。
