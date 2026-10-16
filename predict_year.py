# -*- coding: utf-8 -*-
# @Author: WeiweiDeng
# @Date: 2026/2/6 14:27
# @File: predict_year.py
# @Desc: Train LSTM feature models and evaluate a 52-week coal price forecast.
# @Version: 0.0.1

import pandas as pd
import time
from pathlib import Path
import shap
import joblib
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from matplotlib import font_manager
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


# Load a serialized scaler.
def stand_s(model):
    with open(model, 'rb') as f:
        m_x = pickle.load(f)
    return m_x


# Build weekly windows: 52 weeks of input predict the following 52 weeks.
def create_weekly_avg_dataset(series, look_back=52, pred_horizon=52, week=7):
    X_7, x, y = [], [], []
    for i in range(0, len(series), week):
        X_7.append(series[i:i + week, 0].mean())
    print(f"Observed weekly values: {len(X_7)}")
    for j in range(len(X_7) - look_back - pred_horizon + 1):
        x.append(X_7[j: j+look_back])
        y.append(X_7[j+look_back: j+look_back+pred_horizon])
    # Return weekly means, lookback windows, and forecast target windows.
    return np.array(X_7), np.array(x), np.array(y)


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden1=64, hidden2=32, output_size=52):
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

columns = [col for col in data.columns if col != target_column]

real = data[target_column].values
# Observed weekly average prices
real_avg = []
for i in range(0, len(real), 7):
    real_avg.append(real[i:i + 7].mean().item())

last_price = real_avg[-53]  # Start recursive prediction with the price preceding the final 52 weeks.

final_predict = np.zeros((52, len(columns)))  # Store each predicted feature in its column; reserve the last for historical price.
evaluation = {}  # Collect evaluation metrics.

start = time.time()
print(f"================Training started================")
for idx, col in enumerate(columns):
    feature_label = f"Feature {idx}"
    # Split the data before fitting scalers.
    # Build inputs and targets using the configured sliding windows.
    # Build sliding windows from the observed data.
    look_back = 52  # Use 104 for a two-year lookback.
    total_data, X, y = create_weekly_avg_dataset(data[col].values.reshape(-1, 1))

    # Hold out the final window for evaluation.
    train_size = len(X) - 1
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    # Scale each feature independently, then restore predictions before SVR preprocessing.
    # Flatten the input windows into one column for scaling.
    # Fit once, then restore the original window shape.
    shape_tr = X_train.shape
    shape_te = X_test.shape

    X_train_flat = X_train.reshape(-1, 1)
    X_test_flat = X_test.reshape(-1, 1)

    scale_x = StandardScaler()
    X_train_flat = scale_x.fit_transform(X_train_flat)
    X_test_flat = scale_x.transform(X_test_flat)

    X_train = X_train_flat.reshape(shape_tr)  # Restore the window shape.
    X_test = X_test_flat.reshape(shape_te)

    scale_y = StandardScaler()
    y_train = scale_y.fit_transform(y_train)
    y_test = scale_y.transform(y_test)

    # Reshape the scaled feature for the LSTM.
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    print("Test input shape:", X_test.shape)

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

        print(f"Epoch {epoch + 1} Loss {epoch_loss / len(loader):.4f}")

    torch.save(model.state_dict(), f'./lstm_models/year/lstm_{idx}.pt')

    print(f'{feature_label} model saved')

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
    y_pre = scale_y.inverse_transform(future_preds).ravel()
    # Store predictions in their original units for later SVR preprocessing.
    final_predict[:, idx] = y_pre

    # Save the model.
    name = f'./model/year/{idx}_input_x_StandardScaler.pkl'
    with open(name, 'wb') as file:
        pickle.dump(scale_x, file)
    name = f'./model/year/{idx}_input_y_StandardScaler.pkl'
    with open(name, 'wb') as file:
        pickle.dump(scale_y, file)

    # print("Actual test values:", y_test)

    # mae = mean_absolute_error(y_test.ravel(), y_pre)
    # mse = mean_squared_error(y_test.ravel(), y_pre)
    # rmse = np.sqrt(mse)
    # r2 = r2_score(y_test.ravel(), y_pre)
    # print(f'{feature_label} - Test MAE: {mae: .2f}, RMSE: {rmse: .2f}, R2: {r2: .2f}')
    # evaluation[feature_label] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}

    a = range(len(total_data))  # Index weekly observations from zero.
    forecast_index = range(len(total_data)-52, len(total_data))  # Indices for the final 52 observed weeks
    forecast = pd.Series(y_pre.flatten(), index=forecast_index)

    print(f"Forecast for the final 52 weekly values of {feature_label}:")
    print(forecast)

    # Plot actual and predicted values.
    plt.figure(figsize=(10, 4))
    plt.plot(a[-80:], total_data[-80:], label='Actual')
    plt.plot(forecast_index, forecast.values, marker='o', label='Predicted')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./result/Images/year/'+f'{idx}forecast.png', dpi=300)
    # plt.show()

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

input_data = np.array(input_data).reshape(52, len(columns))

# Round predicted prices to two decimal places.
price_pre = np.round(price_pre, 2).ravel()
print("Predicted coal prices for 52 weeks:", price_pre)

mae = mean_absolute_error(real_avg[-52:], price_pre)
mse = mean_squared_error(real_avg[-52:], price_pre)
rmse = np.sqrt(mse)
r2 = r2_score(real_avg[-52:], price_pre)
evaluation['Coal price'] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
# Export the forecast evaluation metrics.
eval_result_data = pd.DataFrame(evaluation)
eval_result_datas = eval_result_data.T
eval_result_datas.to_csv('./result/Evaluation/year/year_predict_eva.csv')

plt.figure(figsize=(10, 4))
plt.plot(forecast_index, real_avg[-52:], marker='o', color='black', label='Actual')
plt.plot(forecast_index, price_pre.ravel(), marker='o', color='red', label='Predicted')
plt.title(f"52-week coal price forecast")
plt.legend()
plt.tight_layout()
plt.savefig('./result/Images/year/price_year.png', dpi=300)

print("=============Feature analysis===============")
explainer = shap.KernelExplainer(svr_model.predict, bg)
shap_values = explainer.shap_values(input_data)
for i in range(len(shap_values)):
    global_imp = pd.Series(np.abs(shap_values[i]), index=[f"Feature {j}" for j in range(len(columns))])
    # global_imp = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_names)
    global_imp.sort_values(ascending=False).to_csv(f"./result/Evaluation/year/{i}_shap.csv")

    # Plot a horizontal bar chart.
    plt.figure(figsize=(12, 8))
    bars = global_imp.sort_values(ascending=True).plot(kind='barh', color='skyblue')  # Sort ascending for the horizontal bar chart.
    plt.xlabel('SHAP Value (Absolute)')
    # plt.title(f'Test sample {i}: actual price {real_avg[i+1]}, predicted price {price_pre[i]: .2f}')
    plt.title(f'Test sample {i}: predicted price {price_pre[i]: .2f}')
    plt.tight_layout()
    # Add a value label at the end of each bar.
    for bar in bars.patches:
        width = bar.get_width()
        plt.annotate(f'{width: .4f}',  # Format the displayed value.
                     (width, bar.get_y() + bar.get_height() / 2),
                     ha='left', va='center')
    plt.savefig(f'./result/Images/year/{i}_shap.png', dpi=300)
    # plt.show()

end = time.time()
print(f"Training completed in {end - start} seconds")
