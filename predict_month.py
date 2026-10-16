# -*- coding: utf-8 -*-
# @Author: WeiweiDeng
# @Date: 2026/2/6 14:38
# @File: predict_month.py
# @Desc: Train LSTM feature models and evaluate a 30-day coal price forecast.
# @Version: 0.0.1

# Soda ash, thermal power generation, and cement are monthly series.
import pandas as pd
import time
from pathlib import Path
import joblib
import numpy as np
import pickle
import matplotlib.pyplot as plt
import shap
from matplotlib import font_manager
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

for output_dir in ['model', 'model/week', 'model/month', 'model/year', 'lstm_models/week', 'lstm_models/month', 'lstm_models/year', 'result/Images', 'result/Images/week', 'result/Images/month', 'result/Images/year', 'result/Evaluation', 'result/Evaluation/year']:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

# Use the optional system font for source-data labels when available.
# font_path = '/System/Library/Fonts/Supplemental/Songti.ttc'
# font_prop = font_manager.FontProperties(fname=font_path)
# plt.rcParams['font.family'] = font_prop.get_name()
# plt.rcParams['axes.unicode_minus'] = False

device = torch.device('cpu')  # Initialize with the CPU device.
look_back = 90
pred_horizon = 30
epochs = 100
batch_size = 32
lr = 0.001

# Load a serialized scaler.
def stand_s(model):
    with open(model, 'rb') as f:
        m_x = pickle.load(f)
    return m_x


# Build daily sliding windows with the configured lookback and horizon.
def create_dataset(data, look_back=90, pred_horizon=30):
    X, y = [], []
    for i in range(len(data) - look_back - pred_horizon + 1):
        X.append(data[i:i+look_back, 0])  # Each row contains one lookback window for this feature.
        y.append(data[i+look_back: i+look_back+pred_horizon, 0])
    return np.array(X), np.array(y)


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden1=64, hidden2=32, output_size=30):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.fc = nn.Linear(hidden2, output_size)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# Load scalers and the pretrained SVR bundle.
input_stand = stand_s('./model/input_StandardScaler.pkl')
output_stand = stand_s('./model/price_StandardScaler.pkl')
bundle = joblib.load('./model/SVR_BG.pkl')
svr_model = bundle['model']
bg = bundle['bg']

data = pd.read_csv('data/cci_jining_coal.csv', index_col=0, encoding='utf-8')
target_column = 'CCI济宁气煤(元/吨)'
last_price = data[target_column].iloc[-30]


columns = [col for col in data.columns if col != target_column]
print("Input feature data:", data[columns].set_axis([f"Feature {i}" for i in range(len(columns))], axis=1))

# data[target_column].index = pd.to_datetime(data[target_column].index)  # Convert before slicing.
data.index = pd.to_datetime(data.index)
start = pd.Timestamp('2025-12-3')
month = data[target_column].loc[start: start + pd.Timedelta(days=30)]
print("Actual 30-day prices:", month.rename("Coal price"))
real = month.values
print(len(real))

final_predict = np.zeros((30, len(columns)))  # Store each predicted feature in its column; reserve the last for historical price.
evaluation = {}  # Collect evaluation metrics.

start = time.time()
print(f"================Training started================")
for idx, col in enumerate(columns):
    feature_label = f"Feature {idx}"
    # Split the data before fitting scalers.
    # Build inputs and targets using the configured sliding windows.
    # Build sliding windows from the observed data.

    look_back = 90

    X, y = create_dataset(data[col].values.reshape(-1, 1), look_back)
    print("Number of windows:", len(X))

    # Hold out the final window for evaluation.
    train_size = len(X) - 1
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    print('X_train shape:', X_train.shape)
    print('y_traint shape:', y_train.shape)
    print('X_test shape:', X_test.shape)
    print('y_test shape:', y_test.shape)

    # Scale each feature independently, then restore predictions before SVR preprocessing.
    # Flatten the input windows into one column for scaling.
    # Fit once, then restore the original window shape.
    shape_tr = X_train.shape
    shape_te = X_test.shape

    X_train_flat = X_train.reshape(-1, 1)
    X_test_flat = X_test.reshape(-1, 1)

    print(X_test_flat.shape)
    scale_x = StandardScaler()
    X_train_flat = scale_x.fit_transform(X_train_flat)
    X_test_flat = scale_x.transform(X_test_flat)

    X_train = X_train_flat.reshape(shape_tr)   # Restore the window shape.
    print(X_train.shape)
    X_test = X_test_flat.reshape(shape_te)

    # Standardize the input and output for this feature.
    # Scaler for target windows
    scale_y = StandardScaler()
    y_train = scale_y.fit_transform(y_train)
    # y_test = scale_y.transform(y_test)

    # Reshape inputs for the LSTM.
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    print("LSTM test input shape:", X_test.shape)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)

    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    print("Training PyTorch LSTM...")

    for epoch in range(100):
        model.train()
        epoch_loss = 0

        for i, (bx, by) in enumerate(loader):
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1} Loss {epoch_loss / len(loader): .4f}")

    torch.save(model.state_dict(), f'./lstm_models/month/lstm_{idx}.pt')

    print(f'{feature_label} model saved')

    print("y_test", y_test)
    last_sequence = X_test_flat
    print("last_sequence", last_sequence)
    model.eval()
    with torch.no_grad():
        inp = torch.tensor(
            last_sequence.reshape(1, look_back, 1),
            dtype=torch.float32
        ).to(device)

        future_preds = model(inp).cpu().numpy()

    # Restore predictions to the original units.
    y_pre = scale_y.inverse_transform(future_preds)
    # Store predictions in their original units for later SVR preprocessing.
    final_predict[:, idx] = y_pre.ravel()
    # Save the fitted scalers.
    name = f'./model/month/{idx}_input_x_StandardScaler.pkl'
    with open(name, 'wb') as file:
        pickle.dump(scale_x, file)
    name = f'./model/month/{idx}_input_y_StandardScaler.pkl'
    with open(name, 'wb') as file:
        pickle.dump(scale_y, file)

    # Evaluate the forecast on the held-out window.
    mae = mean_absolute_error(y_test,  y_pre)
    mse = mean_squared_error(y_test, y_pre)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test.ravel(), y_pre.ravel())
    print(f'{feature_label} - Test MAE: {mae: .2f}, RMSE: {rmse: .2f}, R2: {r2: .2f}')
    evaluation[feature_label] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}

    future_dates = pd.date_range(start='2025-12-2', periods=30, freq='D')  # Daily frequency; periods sets the number of forecast dates.
    forecast = pd.Series(y_pre.flatten(), index=future_dates)
    print(f"30-day forecast for {feature_label}:")
    print(forecast)

    # Use a datetime index for the comparison plot.
    data.index = pd.to_datetime(data.index, errors='coerce')  # Convert the index to datetime.
    plt.figure(figsize=(10, 4))
    plt.plot(data[col].index[-40:], data[col].values[-40:], label='Actual')
    plt.plot(forecast.index, forecast.values, marker='o', label='Predicted')
    plt.title(f"{feature_label} forecast")
    plt.legend()
    plt.tight_layout()
    plt.savefig('./result/Images/month/' + f'{idx}forecast.png', dpi=300)
    # plt.show()

end = time.time()
print(f"Training completed in {end-start} seconds")

# Update the historical price recursively for each forecast step.
h_price_ = []
input_data = []
price_pre = []
for row in final_predict:
    h_price_.append(np.round(last_price, 2))
    row[-1] = last_price
    input_one = input_stand.transform(np.array(row).reshape(1, -1))
    input_data.append(input_one)
    y_pre = svr_model.predict(input_one)
    price_one = output_stand.inverse_transform(y_pre.reshape(-1, 1))
    price_pre.append(price_one)
    last_price = price_one.item()

input_data = np.array(input_data).reshape(30, len(columns))

explainer = shap.KernelExplainer(svr_model.predict, bg)
shap_values = explainer.shap_values(input_data)

price_pre = np.round(price_pre, 2).ravel()
print(f"Predicted coal prices:", price_pre)

# Calculate evaluation metrics.
mae = mean_absolute_error(real, price_pre)
mse = mean_squared_error(real, price_pre)
rmse = np.sqrt(mse)
r2 = r2_score(real, price_pre)
evaluation['Coal price'] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}

# Export the forecast evaluation metrics.
eval_result_data = pd.DataFrame(evaluation)
eval_result_datas = eval_result_data.T
eval_result_datas.to_csv('./result/Evaluation/one_predict_eva.csv')

# Plot the forecast.
plt.figure(figsize=(10, 4))
plt.plot(forecast.index, real, marker='o', color='black', label='Actual')
plt.plot(forecast.index, price_pre.ravel(), marker='o', color='red', label='Predicted')
plt.title(f"Coal price forecast: MAE: {mae: .2f} RMSE: {rmse: .2f} R2: {r2: .2f}")
plt.legend()
plt.tight_layout()
plt.savefig('./result/Images/month/price_price.png', dpi=300)
# plt.show()
