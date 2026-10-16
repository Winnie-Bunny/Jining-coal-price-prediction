# Validation Record

Documentation updated: October 16, 2026.

## Previously completed checks

The following checks were recorded during the initial repository preparation. They do not establish real-model prediction accuracy or validate later changes unless explicitly rerun.

- All project Python files passed syntax parsing.
- All three inference horizons used the same 211 features in the same order. Required LSTMs, per-feature scalers, SVR models, and global scalers were present and nonempty.
- SHA-256 verification passed for 3,823 data and model files.
- Request generation checks passed: model IDs `7`, `30`, and `365` produced 60, 90, and 364 consecutive daily records respectively, each with 211 numeric factors and a numeric historical-price input.
- Flask API tests covered invalid requests returning HTTP 400 without creating a task, unknown tasks returning HTTP 404, successful completion with separate result files, and queryable background errors. Deterministic model substitutes exercised the real routes and background threads; the actual prediction models were not executed.
- A temporary Python 3.13 environment successfully installed the API dependencies in `project_materials_2026-10-16/requirements-test.txt` and passed all four tests.

## Repository preparation changes

- Created log and training-output directories automatically and corrected the callback configuration path.
- Wrote results to a separate file for each task inside the service directory, preventing tasks from overwriting a shared `result.json`.
- Checked model IDs, minimum record counts, and required fields before task creation, preventing invalid requests from leaving tasks in the `running` state.
- Removed the unused `liblinear` import and the `torchvision` and `torchaudio` dependencies. The original `torchvision` and `torch` versions were incompatible.
- Made font configuration portable beyond macOS. Set the service to listen locally by default, with configurable `HOST` and `PORT`, and supplied the corresponding Docker listener setting.
- Added explicit handling for 11 Chinese field aliases in the request generator. Removed one exact duplicate December 31, 2025 record from generated requests while leaving the original CSV unchanged.

## English language update

Documentation and project paths were updated to English. The inference service is under `inference/`, and the dataset is at `data/cci_jining_coal.csv`. Original Chinese data headers and feature-map values remain to preserve compatibility.

After the path changes on October 16, 2026:

- The project integrity check passed, including syntax, feature consistency, required model assets, and all 3,823 SHA-256 digests.
- All four tests passed using the updated paths. API tests continue to use deterministic model substitutes rather than the original trained models.

## Unverified behavior and missing information

- The full PyTorch/SHAP training environment has not been installed for validation. Actual model loading, complete forecasts for all horizons, and retraining have not been executed. Complete files alone do not establish version compatibility or numerical correctness.
- Docker image builds and container startup have not been tested.
- GitHub Actions runs on GitHub after pushes. This record does not claim a successful remote workflow run without verification.
- Data rights and provenance, the original training environment, construction of the historical-price feature, independent backtests, and random seeds still need to be supplied by the original author.
