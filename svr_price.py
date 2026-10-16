# -*- coding: utf-8 -*-
# @Author: WeiweiDeng
# @Date: 2026/2/6 14:40
# @File: svr_price.py
# @Desc: Train the SVR coal price model and analyze feature contributions.
# @Version: 0.0.1
import time
from pathlib import Path
import pickle
import warnings
import shap
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from math import sqrt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # Evaluation metrics
warnings.filterwarnings('ignore')

for output_dir in ['model', 'model/week', 'model/month', 'model/year', 'lstm_models/week', 'lstm_models/month', 'lstm_models/year', 'result/Images', 'result/Images/week', 'result/Images/month', 'result/Images/year', 'result/Evaluation', 'result/Evaluation/year']:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

# Use the optional system font for source-data labels when available.
font_path = Path('/System/Library/Fonts/Supplemental/Songti.ttc')
if font_path.exists():
    font_prop = font_manager.FontProperties(fname=str(font_path))
    plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 8})


def Mape(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)

    # Exclude zero actual values to avoid division by zero.
    mask = actual != 0
    actual = actual[mask]
    predicted = predicted[mask]

    errors = np.abs((actual - predicted) / actual)
    mean_errors = np.mean(errors)
    return mean_errors * 100


def evaluation_process(actual, predicted, modelname):
    mse = mean_squared_error(actual, predicted)
    rmse = sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mape = Mape(actual, predicted)
    r2 = r2_score(actual, predicted)
    evaluation_results = {}
    evaluation_results[modelname] = {
        'MSE': round(mse, 3),
        'RMSE': round(rmse, 3),
        'MAE': round(mae, 3),
        'MAPE': round(mape, 3),
        'R2': round(r2, 3)
    }
    print(evaluation_results)
    return evaluation_results


class SVR_m(object):  # Grid search for the SVR model
    def __init__(self, X_train, y_train, X_test, y_test, scale_y, col):
        self.X_train = X_train
        self.y_train = y_train
        # Test data
        self.X_test = X_test
        self.y_test = y_test
        # Standardization
        self.scale_y = scale_y
        self.feature_names = [f"Feature {i}" for i in range(len(col))]

    def output_model(self):
        SVR()
        param_grid = {
            'kernel': ['linear', 'rbf', 'sigmoid'],  # Kernel types
            'epsilon': [0.01, 0.1, 0.5],
            'C': [0.01, 0.05, 0.1, 0.5, 1.0],  # Regularization parameter candidates
        }

        grid_search = GridSearchCV(SVR(), param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(self.X_train, self.y_train)
        best_model = grid_search.best_estimator_
        print(grid_search.best_params_)

        result_y = best_model.predict(self.X_test)

        # Feature analysis, method 1: permutation importance.
        result = permutation_importance(best_model, self.X_test, self.y_test, n_repeats=10, random_state=42)
        # Visualization
        importances = result.importances_mean
        print(importances)
        plt.figure(figsize=(8, 6))
        plt.bar(range(len(importances)), importances)
        # Color annotations according to the sign of each value.
        # Positive: green; negative: red; near zero: gray.
        colors = []
        for value in importances:
            if value > 0.001:  # Positive value above the threshold
                colors.append('green')
            elif value < -0.001:  # Negative value below the threshold
                colors.append('red')
            else:  # Value near zero
                colors.append('gray')
        # Annotate each bar with its value.
        for i, value in enumerate(importances):
            # Position the annotation.
            y_pos = value
            ha = 'center'
            va = 'bottom' if value >= 0 else 'top'

            # Use a darker shade of the corresponding bar color.
            text_color = 'darkgreen' if value > 0.001 else 'darkred' if value < -0.001 else 'black'

            plt.text(i, y_pos, f'{value: .4f}',
                     ha=ha, va=va, fontsize=7, fontweight='bold',
                     color=text_color, rotation=0)
        plt.xticks(range(len(importances)), self.feature_names, rotation=45, ha='right')
        plt.ylabel("Importance")
        plt.title("Permutation Feature Importance (SVR)")
        plt.savefig('./result/Images/importance_svr.png', dpi=400)
        plt.show()

        p_y = self.scale_y.inverse_transform(result_y.reshape(-1, 1))
        # p_y = result_y.reshape(-1, 1)
        t_y = self.y_test[:, 0]
        # Evaluation metrics
        eval_result = evaluation_process(t_y, p_y, 'SVR')
        # Plot the results.
        plt.figure(figsize=(16, 6))
        plt.plot(t_y, '-o', label='Actual')
        plt.plot(p_y, '-*', label='Predicted')
        plt.legend()
        plt.title('SVR')
        plt.title(
                f"model name: SVR"
                f"\nMAPE: {eval_result['SVR']['MAPE']} %"
                f"R2: {eval_result['SVR']['R2']}")
        plt.savefig('./result/Images/price_svr.png', dpi=400)
        plt.show()

        # Feature analysis, method 2: SHAP values.
        background = self.X_train[-95:, :]
        explainer = shap.KernelExplainer(best_model.predict, background)
        shap_values = explainer.shap_values(self.X_test)

        # pos_mean = np.clip(shap_values, 0, None).mean(axis=0)
        # neg_mean = np.clip(shap_values, None, 0).mean(axis=0)  # Use abs for magnitude.
        # pd.DataFrame({'positive': pos_mean,
        #                   'negative': neg_mean,
        #                   'net': pos_mean + neg_mean},  # Directional net effect
        #                  index=feature_names).to_csv('./result/pos_neg_net_svr.csv')
        # print("SHAP values for test data:", shap_values)

        for i in range(len(shap_values)):
            global_imp = pd.Series(np.abs(shap_values[i]), index=self.feature_names)
            global_imp.sort_values(ascending=False).to_csv(f"./result/test_sample_{i}_shap_svr.csv")

            # Plot a horizontal bar chart.
            plt.figure(figsize=(12, 8))
            bars = global_imp.sort_values(ascending=True).plot(kind='barh', color='skyblue')  # Sort ascending for the horizontal bar chart.
            plt.xlabel('SHAP Value (Absolute)')
            plt.title(f'Test sample {i}: actual price {t_y[i]}, predicted price {p_y[i][0]: .2f}')
            plt.tight_layout()
            # Add a value label at the end of each bar.
            for bar in bars.patches:
                width = bar.get_width()
                plt.annotate(f'{width: .6f}',  # Format the displayed value.
                             (width, bar.get_y() + bar.get_height() / 2),
                             ha='left', va='center')
            plt.savefig(f'./result/test_sample_{i}_shap_svr.png', dpi=300)
            # plt.show()

        # Visualization
        X_test_df = pd.DataFrame(self.X_test, columns=self.feature_names)
        shap.summary_plot(shap_values, X_test_df, show=False)
        plt.savefig('./result/Images/price_shap_svr.png', dpi=400)
        plt.close()
        # Save the model.
        joblib.dump({'model': best_model, 'bg': background},  # Include the SHAP background data.
                    './model/SVR_BG.pkl')
        return eval_result


if __name__ == '__main__':
    start = time.time()
    # Load the data.
    data = pd.read_csv('data/cci_jining_coal.csv', index_col=0, encoding='utf-8')
    col = data.columns.tolist()

    data_LR = data[col]
    # Separate input features and target values.
    input_LR = data_LR.iloc[:, :-1].values
    output_LR = data_LR.iloc[:, -1].values.reshape(-1, 1)

    # Split chronologically without shuffling.
    a = len(input_LR) - 67  # Retain the existing 67-row holdout offset for the weekly setup.
    b = a + 1
    X_train = input_LR[:a, :]
    X_test = input_LR[b:, :]
    y_train = output_LR[:a, :]
    y_test = output_LR[b:, :]

    # Standardization
    scale_x = StandardScaler()
    X_train = scale_x.fit_transform(X_train)
    X_test = scale_x.transform(X_test)

    # Save the model.
    path_x = './model/input_StandardScaler.pkl'
    with open(path_x, 'wb') as file:
        # Serialize the input scaler with pickle.
        # Reload this scaler for consistent preprocessing during inference.
        pickle.dump(scale_x, file)

    scale_y = StandardScaler()
    y_train = scale_y.fit_transform(y_train)
    path_y = './model/price_StandardScaler.pkl'
    with open(path_y, 'wb') as file:
        pickle.dump(scale_y, file)

    # Train the model.
    print("===========SVR============")
    eval_result = {}
    svc_m = SVR_m(X_train, y_train, X_test, y_test, scale_y, col[:-1])
    SVC_eval_result = svc_m.output_model()
    eval_result.update(SVC_eval_result)

    # Export the evaluation metrics.
    eval_result_data = pd.DataFrame(eval_result)
    eval_result_datas = eval_result_data.T
    eval_result_datas.to_csv('./result/Evaluation/Eval_results.csv')
    end = time.time()
    print("Training time (seconds):", end-start)
    print("Finish")
