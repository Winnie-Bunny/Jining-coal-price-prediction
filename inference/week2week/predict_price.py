import joblib
import pickle
import shap
import collections
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

MODEL_DIR = Path(__file__).with_name('model')
LSTM_DIR = Path(__file__).with_name('lstm_models')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the scalers and regression model
def stand_s(model):
    with open(model, 'rb') as f:
        m_x = pickle.load(f)
    return m_x


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden1=64, hidden2=32, output_size=7):
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

    def predict(self, x):
        with torch.no_grad():
            # Reshape the input: (1, look_back, 1) -> [batch_size, seq_len, input_size]
            inp = torch.tensor(
                x.reshape(1, 60, 1),
                dtype=torch.float32
            ).to(device)
            # Run the forward pass and convert the output to a NumPy array on the CPU
            future_preds = self(inp).cpu().numpy()
        return future_preds


print("=======================Set up feature fields===========================")
FEATURE_MAP = {
    # 0-10
    "AHHB_13J": "安徽淮北1/3焦（元/吨）",
    "AHHB_ZJ": "安徽淮北主焦（元/吨）",
    "GQMD_M5YM": "甘其毛都蒙5#原煤（元/吨）",
    "GZPX_ZJM": "贵州盘县主焦煤（元/吨）",
    "HBHD_ZJ": "河北邯郸主焦（元/吨）",
    "HBTS_FM": "河北唐山肥煤（元/吨）",
    "HBTS_MGZJ": "河北唐山蒙古主焦（元/吨）",
    "HNPDS_ZJ": "河南平顶山主焦（元/吨）",
    "HLJHG_13J": "黑龙江鹤岗1/3焦（元/吨）",
    "JQ_M5JM": "金泉蒙5#精煤（元/吨）",
    "JTG_2XZJ": "京唐港二线主焦（元/吨）",
    "JTG_1XZJ": "京唐港一线主焦（元/吨）",
    "NMGWH_FM": "内蒙古乌海肥煤（元/吨）",
    "QHTJ_ZJ": "青海天峻主焦（元/吨）",
    "XJAKS_13J": "新疆阿克苏1/3焦（元/吨）",
    "GZPX_PCM": "贵州盘县喷吹煤（元/吨）",
    "NXSZS_PCM": "宁夏石嘴山喷吹煤（元/吨）",
    "AY_ZY1GXM": "安阳准一级干熄焦（元/吨）",
    "HD_ZY1GXM": "邯郸准一级干熄焦（元/吨）",
    "TS_ZY1GXM": "唐山准一级干熄焦（元/吨）",
    # 20-30
    "AHTL_5000": "安徽铜陵5000（元/吨）",
    "CK_MKAYM": "策克马克A原煤（元/吨）",
    "HBTS_2J": "河北唐山二级焦（元/吨）",
    "HBTS_ZY1J": "河北唐山准一级焦（元/吨）",
    "HNPDS_2J": "河南平顶山二级焦（元/吨）",
    "NMGWH_2J": "内蒙古乌海二级焦（元/吨）",
    "YNKM_2J": "云南昆明二级焦（元/吨）",
    "JZ_PCM": "焦作喷吹煤（元/吨）",
    "BT_JJM": "包头焦精煤（元/吨）",
    "DLG_13JM_E": "大连港1/3焦煤(俄）（元/吨）",
    "DLG_13JM_HG": "大连港1/3焦煤(鹤岗)（元/吨）",
    "DWK_JJM": "大武口焦精煤（元/吨）",
    "FCG_13JM_A": "防城港1/3焦煤(澳)（元/吨）",
    "FS_QM": "抚顺气煤（元/吨）",
    "FK_PCM": "阜康喷吹煤（元/吨）",
    "FK_QM": "阜康气煤（元/吨）",
    "FY_13JJM": "富源1/3焦精煤（元/吨）",
    "GQMD_MGYM": "甘其毛都蒙古原煤（元/吨）",
    "GQMD_MGZJ": "甘其毛都蒙古主焦（元/吨）",
    "GQMD_ZJJM_MG": "甘其毛都主焦精煤（蒙）（元/吨）",
    # 40-50
    "GQMD_ZJYM_MG": "甘其毛都主焦原煤（蒙）（元/吨）",
    "CK_KA_13GLJM": "港口：策克口岸策克1/3高硫焦煤(含税）（元/吨）",  # Shortened display name
    "CK_KA_13JJM": "港口：策克口岸策克1/3焦精煤（元/吨）",
    "CK_KA_13JM_MG": "港口：策克口岸策克1/3焦煤（蒙古）（元/吨）",
    "CK_KA_13JM_YM": "港口：策克口岸策克1/3焦煤（原煤）（元/吨）",
    "CK_KA_13JM_YMHS": "港口：策克口岸策克1/3焦煤（原煤含税）（元/吨）",
    "HM_QM": "哈密气煤（元/吨）",
    "HD_FJM": "邯郸肥精煤（元/吨）",
    "HD_JJM": "邯郸焦精煤（元/吨）",
    "HD_PCM": "邯郸喷吹煤（元/吨）",
    "HD_PSM": "邯郸贫瘦煤（元/吨）",
    "HC_PCM": "韩城喷吹煤（元/吨）",
    "HC_SJM": "韩城瘦精煤（元/吨）",
    "HBKL_ZJM": "河北开滦集团主焦煤（元/吨）",
    "HJ_SM": "河津瘦煤（元/吨）",
    "HB_PSM": "鹤壁贫瘦精煤（元/吨）",
    "HG_13JJM": "鹤岗1/3焦精煤（元/吨）",
    "HG_QM": "鹤岗气煤（元/吨）",
    "HB_13JJM": "淮北1/3焦精煤（元/吨）",
    "HB_FJJM": "淮北肥精煤（元/吨）",
    # 60-70
    "HB_JJM": "淮北焦精煤（元/吨）",
    "HB_SJJM": "淮北瘦精煤（元/吨）",
    "HN_13J": "淮南1/3焦（元/吨）",
    "HZ_13JM": "霍州1/3焦煤（元/吨）",
    "HZ_FM_CB": "霍州肥煤(车板)（元/吨）",
    "HZ_FM_KK": "霍州肥煤(坑口)（元/吨）",
    "JX_13JJM": "鸡西1/3焦精煤（元/吨）",
    "JX_FJM": "鸡西肥精煤（元/吨）",
    "JX_JJM": "鸡西焦精煤（元/吨）",
    "JZ_WYMM": "焦作无烟末煤（元/吨）",
    "JTG_AZ_13J": "京唐港澳洲1/3焦（元/吨）",
    "JTG_AZ_ZJM": "京唐港澳洲主焦煤（元/吨）",
    "JTG_EL_13J": "京唐港俄罗斯1/3焦（元/吨）",
    "JTG_EL_ZJM": "京唐港俄罗斯主焦煤（元/吨）",
    "JTG_HB_ZJM": "京唐港河北主焦煤（元/吨）",
    "JTG_MG_ZJM": "京唐港美国主焦煤（元/吨）",
    "JTG_NM_13J": "京唐港内蒙1/3焦（元/吨）",
    "JTG_NM_FM": "京唐港内蒙古肥煤（元/吨）",
    "JTG_SX_13JM": "京唐港山西1/3焦煤（元/吨）",
    "JTG_SX_ZJM": "京唐港山西主焦煤（元/吨）",
    # 80-90
    "JTG_SM_J": "京唐港瘦煤(加)（元/吨）",
    "JTG_ZJM_J": "京唐港主焦煤(加)（元/吨）",
    "KL_13JJM": "开滦1/3焦精煤（元/吨）",
    "KL_FJM": "开滦肥精煤（元/吨）",
    "KL_JJM": "开滦焦精煤（元/吨）",
    "KM_13JJM": "昆明1/3焦精煤（元/吨）",
    "LYG_QFM_MG": "连云港气肥煤(美国)（元/吨）",
    "LYG_SM_E": "连云港瘦煤(俄)（元/吨）",
    "LYG_ZJM_A": "连云港主焦煤(澳)（元/吨）",
    "LYG_ZJM_SX": "连云港主焦煤(山西)（元/吨）",
    "LPS_13JJM": "六盘水1/3焦精煤（元/吨）",
    "LPS_JJM_G80P": "六盘水焦精煤(G>80)（元/吨）",
    "LPS_JJM_G80": "六盘水焦精煤(G80)（元/吨）",
    "LPS_JJM_G86": "六盘水焦精煤(G86)（元/吨）",
    "LPS_SJM": "六盘水瘦精煤（元/吨）",
    "LMHG_13JM_A85": "龙煤鹤岗1/3焦煤(A8.5)（元/吨）",
    "LMHG_13JM_A95": "龙煤鹤岗1/3焦煤(A9.5)（元/吨）",
    "LM_JM": "龙煤集团焦煤（元/吨）",
    "LC_PCM": "潞城喷吹煤（元/吨）",
    "NW_13JM": "宁武1/3焦煤（元/吨）",
    # 100-110
    "PZH_JJM": "攀枝花焦精煤（元/吨）",
    "PDS_13JM": "平顶山1/3焦煤（元/吨）",
    "PDS_ZJ": "平顶山主焦（元/吨）",
    "PL_SM": "平陆瘦煤（元/吨）",
    "PX_13JJM": "萍乡1/3焦精煤（元/吨）",
    "PX_JJM": "萍乡焦精煤（元/吨）",
    "QTH_13JJM": "七台河1/3焦精煤（元/吨）",
    "QTH_FJM": "七台河肥精煤（元/吨）",
    "QTH_JJM": "七台河焦精煤（元/吨）",
    "SHG_ZJM_A": "上海港主焦煤(澳)（元/吨）",
    "SJZ_JJM": "石家庄焦精煤（元/吨）",
    "SJZ_PCM": "石家庄喷吹煤（元/吨）",
    "SZS_FJM": "石嘴山肥精煤（元/吨）",
    "SZS_PSM": "石嘴山贫瘦煤（元/吨）",
    "SZS_ZJM": "石嘴山主焦煤（元/吨）",
    "SY_PCM": "寿阳喷吹煤（元/吨）",
    "SYS_13JJM": "双鸭山1/3焦精煤（元/吨）",
    "SYS_QJM": "双鸭山气精煤（元/吨）",
    "SZ_13JM": "宿州1/3焦煤（元/吨）",
    "TS_FJM": "唐山肥精煤（元/吨）",
    "TS_JJM": "唐山焦精煤（元/吨）",
    # 120-130
    "TJG_AZ_13J": "天津港澳洲1/3焦（元/吨）",
    "TJG_AZ_ZJM": "天津港澳洲主焦煤（元/吨）",
    "TJG_E_QFM": "天津港俄罗斯气肥煤（元/吨）",
    "TJG_E_ZJM": "天津港俄罗斯主焦煤（元/吨）",
    "TJG_MG_13JM": "天津港美国1/3焦煤（元/吨）",
    "TJG_MG_13JM_A8": "天津港美国1/3焦煤(A8)（元/吨）",
    "TJG_MG_ZJM": "天津港美国主焦煤（元/吨）",
    "TJG_YN_QM": "天津港印尼气煤（元/吨）",
    "TJG_ZJM_MG": "天津港主焦煤(蒙古)（元/吨）",
    "TJ_13JM": "天峻1/3焦煤（元/吨）",
    "TJ_JM_G70": "天峻焦煤(G70)（元/吨）",
    "TJ_JM_G80": "天峻焦煤(G80)（元/吨）",
    "TH_JJM": "通化焦精煤（元/吨）",
    "TL_PCM": "屯留喷吹煤（元/吨）",
    "TL_PSM": "屯留贫瘦煤（元/吨）",
    "WH_13JJM": "乌海1/3焦精煤（元/吨）",
    "WH_FJM_G88": "乌海肥精煤(G>88)（元/吨）",
    "WH_FJM_G90": "乌海肥精煤(G90)（元/吨）",
    "WLMQ_13JM": "乌鲁木齐1/3焦煤（元/吨）",
    "WLMQ_QM": "乌鲁木齐气煤（元/吨）",
    "WLMQ_RNM": "乌鲁木齐弱粘煤（元/吨）",
    # 140-150
    "WW_13J": "武威1/3焦（元/吨）",
    "XN_10JM": "乡宁10号焦煤（元/吨）",
    "XN_2JM": "乡宁2号焦煤（元/吨）",
    "XY_SJM": "襄垣瘦精煤（元/吨）",
    "XY_SM": "襄垣瘦煤（元/吨）",
    "XY_JM": "孝义焦煤（元/吨）",
    "XY_JM_G80": "孝义焦煤(G80)（元/吨）",
    "XT_13JJM": "邢台1/3焦精煤（元/吨）",
    "XT_JJM": "邢台焦精煤（元/吨）",
    "XX_QM": "兴县气煤（元/吨）",
    "XZ_QJM": "徐州气精煤（元/吨）",
    "XC_SJM": "许昌瘦精煤（元/吨）",
    "XW_JJM": "宣威焦精煤（元/吨）",
    "YNMG_CK_13JM": "冶金煤:内蒙古策克1/3焦煤（元/吨）",
    "YKG_ZJM_A": "营口港主焦煤(澳)（元/吨）",
    "YKG_ZJM_HLJ": "营口港主焦煤(黑龙江)（元/吨）",
    "YC_PCM": "永城喷吹煤（元/吨）",
    "YC_13JJM": "永川1/3焦精煤（元/吨）",
    "YC_FJM": "永川肥精煤（元/吨）",
    "CQ_JJM": "重庆焦精煤（元/吨）",
    "CQ_SJM": "重庆瘦精煤（元/吨）",
    # 160-170 (Macroeconomic and futures indicators)
    "SH_LJMKC": "全社会炼焦煤库存（万吨）",
    "QH_JT_PRICE": "焦炭主力合约收盘价（元/吨）",
    "QH_TKS_PRICE": "铁矿石主力合约收盘价（元/吨）",
    "QH_LWG_PRICE": "螺纹钢主力合约收盘价（元/吨）",
    "QH_RBK_PRICE": "热轧卷板主力合约收盘价（元/吨）",
    "NC_YM_CL_D": "全国原煤产量当月值（万吨）",
    "NC_YM_CL_L": "全国原煤产量累计值（万吨）",
    "GP_TS_PRICE": "钢坯(唐山)价格（元/吨）",
    "ZB_JL_LYL": "焦炉产能利用率%",
    "KC_JT": "焦炭库存（万吨）",
    "LP_LJM_LPL": "炼焦煤竞拍流拍率",
    "KC_LJM_DAYS": "炼焦煤库存可用天数",
    "KC_LJM_GC": "炼焦煤下游钢厂库存（万吨）",
    "KC_LJM_JH": "炼焦煤下游焦化厂库存（万吨）",
    "JG_RBK_SH": "热轧板卷（上海）价格（元/吨）",
    "JG_LWG_HZ": "热轧螺纹钢（杭州）价格（元/吨）",
    "JG_TKS": "铁矿石价格（元/吨）",
    "CL_TS": "铁水产量（万吨）",
    "RZG_JK_PCM": "日照港进口喷吹煤（元/吨）",  # Shortened Chinese display name
    "SDRZ_ZY1J": "山东日照准一级焦（元/吨）",
    # 180-190
    "SDRZ_2J": "山东日照二级焦（元/吨）",
    "LY_ZY1GXM": "临沂准一级干熄焦（元/吨）",
    "CCI_JNQMJ": "CCI济宁气煤周均价（元/吨）",
    "JN_QFM": "济宁气肥煤（元/吨）",
    "JN_QHM": "济宁气混煤（元/吨）",
    "LY_QM": "临沂气煤（元/吨）",
    "QDG_ZJM_A": "青岛港主焦煤(澳)（元/吨）",
    "RZG_FM_LF": "日照港肥煤(临汾)（元/吨）",
    "RZG_LJM_JD": "日照港炼焦煤(加)（元/吨）",
    "RZG_PCM_JC": "日照港喷吹煤(晋城)（元/吨）",
    "RZG_QM_MG": "日照港气煤(美国)（元/吨）",
    "RZG_SM_EL": "日照港瘦焦煤(俄)（元/吨）",
    "RZG_ZJM_A": "日照港主焦煤(澳)（元/吨）",
    "TA_QJM": "泰安气精煤（元/吨）",
    "TS_13JJM": "唐山1/3焦精煤（元/吨）",
    "TZ_JJM": "滕州焦精煤（元/吨）",
    "TZ_QFM": "滕州气肥煤（元/吨）",
    "YZ_QHM": "兖州气混煤（元/吨）",
    "YZ_QJM_A10": "兖州气精煤(A10)（元/吨）",
    "YZ_QJM_A8": "兖州气精煤(A8)（元/吨）",
    # 200-211
    "YZ_QJM_A9": "兖州气精煤(A9)（元/吨）",
    "ZZ_13JM": "枣庄1/3焦煤（元/吨）",
    "ZZ_FJM": "枣庄肥精煤（元/吨）",
    "ZZ_QJM": "枣庄气精煤（元/吨）",
    "ZB_QM": "淄博气煤（元/吨）",
    "CCI_JNQMX": "CCI济宁气煤旬均价（元/吨）",
    "CCI_JNQMY": "CCI济宁气煤月均价（元/吨）",
    "RZG_KC_JT": "日照港焦炭库存量(万吨)"
}

FEATURE_KEYS = list(FEATURE_MAP.keys())
FEATURE_NAMES_CN = list(FEATURE_MAP.values())

print(f"Loaded {len(FEATURE_KEYS)} feature mappings. Example: {FEATURE_KEYS[0]} -> {FEATURE_NAMES_CN[0]}")

print("=======================Load the regression model=============================")
input_svr_stand = stand_s(MODEL_DIR/'input_StandardScaler.pkl')
output_svr_stand = stand_s(MODEL_DIR/'price_StandardScaler.pkl')
bundle = joblib.load(MODEL_DIR/'SVR_BG.pkl')
svr_model = bundle['model']
bg = bundle['bg']
print("=======================Load LSTM models=============================")
# Load all models dynamically
input_x_models = {}
input_y_models = {}
lstm_models = {}

print("Loading model files...")
for idx, key in enumerate(FEATURE_KEYS):
    # Filename format: 0_input_x..., lstm_0.pt
    file_x = f"{idx}_input_x_StandardScaler.pkl"
    file_y = f"{idx}_input_y_StandardScaler.pkl"
    file_lstm = f"lstm_{idx}.pt"

    try:
        input_x_models[key] = stand_s(MODEL_DIR / file_x)
        input_y_models[key] = stand_s(MODEL_DIR / file_y)

        # Load the PyTorch model
        # Initialize the model with the same parameters used for training
        model = LSTMModel(input_size=1, hidden1=64, hidden2=32, output_size=7).to(device)

        model.load_state_dict(torch.load(LSTM_DIR / file_lstm, map_location='cpu', weights_only=True))
        model.to(device)
        model.eval()
        lstm_models[key] = model
    except Exception as e:
        print(f"Error: failed to load feature [{key}] (index {idx}): {e}")
        raise e

print("Model loading complete")


# Read input data
def get_data(data):

    input_data = data['price_data']  # Raw coal price data
    price_factor = {}  # Price factors
    history_price = []  # Previous coal price
    for day in input_data:
        day_features = {}
        for key in FEATURE_KEYS:
            day_features[key] = day[key]
        day_features["history_price"] = day["history_price"]
        price_factor[day["date"]] = day_features
        history_price.append(day["history_price"])

    price_factor_df = pd.DataFrame(price_factor)
    price_data = price_factor_df.T
    # Keep the latest input date to assign dates to the forecast
    date_last = price_data.index[-1]
    history_price = np.array(history_price)

    print(price_data)
    return price_data, history_price, date_last


print("=======================Start inference=============================")


def optimize_model(price_data, history_price, date_last):
    print("Start inference")
    result_dict = []
    columns = price_data.columns
    look_back = 60
    print("Output column names:", columns)
    final_predict = np.zeros((7, 212))  # Store each forecast in its corresponding column; reserve the final column for the previous price
    for idx, col in enumerate(columns[:-1]):  # Skip forecasting the previous-price column
        model_x = input_x_models[col]
        model_y = input_y_models[col]
        X = price_data[col].values[-look_back:]
        X = X.reshape(-1, 1)
        X_flat = model_x.transform(X)
        # print(X_flat)
        # print("Shape after scaling:", X_flat.shape)
        X_stand = X_flat.reshape(1, look_back, 1)
        # print("LSTM input shape:", X_stand.shape)
        lstm_model = lstm_models[col]
        future_pre = lstm_model.predict(X_stand)
        y_pre = model_y.inverse_transform(future_pre)
        final_predict[:, idx] = y_pre.ravel()

    new_price = history_price[-1]
    h_price_ = []
    input_data = []
    price_pre = []
    for row in final_predict:
        h_price_.append(np.round(new_price, 2))
        row[-1] = new_price
        input_one = input_svr_stand.transform(np.array(row).reshape(1, -1))
        input_data.append(input_one)
        y_pre = svr_model.predict(input_one)
        price_one = output_svr_stand.inverse_transform(y_pre.reshape(-1, 1))
        price_pre.append(price_one)
        new_price = price_one.item()

    print("Previous prices used for prediction:", h_price_)
    price_pre = np.array(price_pre)
    price_pre = np.round(price_pre, 2)
    print("Predicted coal prices for seven days:", price_pre)
    price_result = []
    for item in price_pre:
        if isinstance(item, np.ndarray):
            price_result.append(item.item())  # Extract a scalar from a NumPy array with item()
        else:
            price_result.append(item)  # Append other types, such as np.int64, directly

    h_price_result = []
    for item in h_price_:
        if isinstance(item, np.ndarray):
            h_price_result.append(item.item())  # Extract a scalar from a NumPy array with item()
        else:
            h_price_result.append(item)  # Append other types, such as np.int64, directly

        # Create forecast dates
    start = pd.Timestamp(date_last)
    n = 7  # Use the next seven days, excluding the starting date
    dates = pd.date_range(start + pd.Timedelta(days=1), periods=n)

    input_data = np.array(input_data).reshape(7, 212)
    print("=============Feature analysis===============")
    explainer = shap.KernelExplainer(svr_model.predict, bg)
    shap_values = explainer.shap_values(input_data)
    for i in range(len(shap_values)):
        global_imp = pd.Series(np.abs(shap_values[i]), index=columns)
        global_imp_sorted = global_imp.sort_values(ascending=False)
        result_dict.append({
            'date': dates[i].strftime('%Y-%m-%d'),
            'predict_history_price': h_price_result[i],
            'predict_price': price_result[i],
            'factor_analysis': collections.OrderedDict(global_imp_sorted)
        })
        for idx, key in enumerate(FEATURE_KEYS):
            result_dict[i][f"predict_{key}"] = round(float(final_predict[i][idx]), 2)
    return result_dict



