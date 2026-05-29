import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

SEED = 42

# ====================== 【完全匹配你的工艺】 ======================
DV_CONFIG = {
    "inflow_cod": -4.5,
    "inflow_nh3": -4.5,
    "inflow_rate2": -0.05,
    "bio2_mlss_meas": 0.0
}

MV_COLS = [
    "bio2_front_aeration_flow_meas",
    "bio2_end_aeration_flow_meas"
]

CV_CONFIG = {
    "bio2_end_do_meas": 0.5,
    "outflow_cod": 8.0,
    "outflow_nh3": 8.0
}

def hours_to_steps(hours, sample_interval_min):
    return int(round(hours * 60 / sample_interval_min))

# ====================== 特征工程（使用真实字段） ======================
def build_features(df_input, sample_interval_min):
    df = df_input.copy()
    dff = df.copy()

    # 1. DV 滞后
    for col, offset in DV_CONFIG.items():
        steps = hours_to_steps(abs(offset), sample_interval_min)
        dff[f"{col}_lag{steps}"] = df[col].shift(steps)

    # 2. 曝气总量
    dff[MV_COLS] = df[MV_COLS]
    dff["MV_total"] = df[MV_COLS].sum(axis=1)

    # 3. 当前状态
    dff[list(CV_CONFIG.keys())] = df[list(CV_CONFIG.keys())]

    # 4. DO 滞后
    for h in [0.5, 1, 2, 3]:
        s = hours_to_steps(h, sample_interval_min)
        dff[f"bio2_end_do_meas_lag{h}h"] = df["bio2_end_do_meas"].shift(s)

    # 5. 曝气滞后
    for h in [0.5, 1, 2]:
        s = hours_to_steps(h, sample_interval_min)
        dff[f"MV_total_lag{h}h"] = dff["MV_total"].shift(s)

    # 6. DO 滚动均值
    for h in [1, 3, 6]:
        s = hours_to_steps(h, sample_interval_min)
        dff[f"do_roll_mean_{h}h"] = df["bio2_end_do_meas"].shift(1).rolling(s).mean()

    # 7. 进水滚动均值
    for h in [4, 8, 12, 24]:
        s = hours_to_steps(h, sample_interval_min)
        dff[f"inflow_cod_mean_{h}h"] = df["inflow_cod"].shift(1).rolling(s).mean()
        dff[f"inflow_nh3_mean_{h}h"] = df["inflow_nh3"].shift(1).rolling(s).mean()

    # 8. 长周期特征
    dff["aeration_sum_8h"] = dff["MV_total"].shift(1).rolling(hours_to_steps(8, sample_interval_min)).sum()
    dff["aeration_sum_24h"] = dff["MV_total"].shift(1).rolling(hours_to_steps(24, sample_interval_min)).sum()
    dff["mlss_mean_8h"] = df["bio2_mlss_meas"].shift(1).rolling(hours_to_steps(8, sample_interval_min)).mean()
    dff["mlss_mean_24h"] = df["bio2_mlss_meas"].shift(1).rolling(hours_to_steps(24, sample_interval_min)).mean()

    # 9. 出水滞后
    for h in [2, 4, 8, 12, 24]:
        s = hours_to_steps(h, sample_interval_min)
        dff[f"outflow_cod_lag{h}h"] = df["outflow_cod"].shift(s)
        dff[f"outflow_nh3_lag{h}h"] = df["outflow_nh3"].shift(s)

    return dff

def clean_data(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for c in numeric_cols:
        m = df[c].mean()
        s = df[c].std()
        if s > 0:
            df[c] = df[c].clip(m - 3*s, m + 3*s)
    df[numeric_cols] = df[numeric_cols].interpolate(limit=10).ffill().bfill()
    return df

# ====================== 训练模型 ======================
def train_model(df):
    # ======================================
    print("📌 开始训练模型...")
    print(f"📊 原始数据条数：{len(df)}")
    # ======================================

    df = clean_data(df)

    # ======================================
    print(f"✅ 数据清洗完成，剩余：{len(df)} 条")
    # ======================================

    time_diff = df["timestamp"].diff().dropna()
    sample_interval_min = time_diff.dt.total_seconds().median() / 60

    # ======================================
    print(f"⏱ 采样间隔：{sample_interval_min:.2f} 分钟")
    # ======================================

    df_feat = build_features(df, sample_interval_min)

    target_map = {}
    for col, lead in CV_CONFIG.items():
        steps = hours_to_steps(lead, sample_interval_min)
        target = f"{col}_lead{lead}h"
        target_map[col] = target
        df_feat[target] = df[col].shift(-steps)

    df_feat = df_feat.dropna().reset_index(drop=True)

    # ======================================
    print(f"📦 特征工程完成，特征数据量：{df_feat.shape}")
    # ======================================

    feat_cols = [c for c in df_feat.columns if c not in ["timestamp", "id"] + list(target_map.values())]
    print(df_feat)
    n = len(df_feat)
    train = int(0.7 * n)
    val = int(0.15 * n)
    df_train = df_feat.iloc[:train]
    df_val = df_feat.iloc[train:train+val]

    # ======================================
    print(f"📚 训练集：{len(df_train)} 条 | 验证集：{len(df_val)} 条")
    # ======================================

    scaler = RobustScaler()
    model_dict = {}
    selected_feat = {}

    for target_col, target_name in target_map.items():
        # ======================================
        print(f"\n🚀 开始训练目标：{target_col}")
        # ======================================

        y_train = df_train[target_name].values
        X_all = df_train[feat_cols].values
        X_all_s = scaler.fit_transform(X_all)

        selector = SelectFromModel(CatBoostRegressor(random_state=SEED, verbose=0))
        selector.fit(X_all_s, y_train)
        selected = [feat_cols[i] for i in range(len(feat_cols)) if selector.get_support()[i]]

        # ======================================
        print(f"✅ 特征选择完成，选中 {len(selected)} 个特征")
        # ======================================

        X_train = df_train[selected].values
        X_val = df_val[selected].values
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        tscv = TimeSeriesSplit(3)
        grid = GridSearchCV(
            CatBoostRegressor(random_state=SEED, verbose=0),
            {"max_depth": [3,4,5], "learning_rate": [0.04,0.06], "n_estimators": [200,300]},
            cv=tscv
        )
        grid.fit(X_train_s, y_train)
        model = grid.best_estimator_
        model.fit(X_train_s, y_train, eval_set=[(X_val_s, df_val[target_name].values)])

        model_dict[target_col] = model
        selected_feat[target_col] = selected

        # ======================================
        print(f"✅ {target_col} 训练完成！")
        # ======================================

    # 最后保存模型，返回路径
    model_name = f"model_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"
    model_dir = "./models/"
    import os
    os.makedirs(model_dir, exist_ok=True)

    scaler_path = f"{model_dir}{model_name}_scaler.pkl"
    model_path = f"{model_dir}{model_name}_models.pkl"
    config_path = f"{model_dir}{model_name}_config.pkl"

    joblib.dump(scaler, scaler_path)
    joblib.dump(model_dict, model_path)
    joblib.dump({
        "sample_interval_min": sample_interval_min,
        "selected_feat": selected_feat,
        "target_map": target_map
    }, config_path)

    # ======================================
    print("\n🎉 所有模型训练完成！已保存到文件")
    print("📄 scaler.pkl")
    print("📄 models.pkl")
    print("📄 model_config.pkl\n")
    # ======================================
    return {
        "model_name": model_name,
        "model_path": model_path,
        "scaler_path": scaler_path,
        "config_path": config_path
    }

# ====================== 预测 ======================
def predict_model(df):
    scaler = joblib.load("scaler.pkl")
    model_dict = joblib.load("models.pkl")
    cfg = joblib.load("model_config.pkl")

    sample_interval_min = cfg["sample_interval_min"]
    selected_feat = cfg["selected_feat"]
    df = clean_data(df)
    df_feat = build_features(df, sample_interval_min)
    df_feat = df_feat.fillna(method="ffill").fillna(method="bfill")

    result = {}
    for target_col in CV_CONFIG.keys():
        feats = selected_feat[target_col]
        X = df_feat[feats].values
        X_s = scaler.transform(X)
        pred = model_dict[target_col].predict(X_s)
        result[target_col] = float(pred[-1])
    return result
