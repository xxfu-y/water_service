import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

SEED = 42

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


# ====================== 特征工程 ======================
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
            df[c] = df[c].clip(m - 3 * s, m + 3 * s)
    df[numeric_cols] = df[numeric_cols].interpolate(limit=10).ffill().bfill()
    return df


# ====================== 训练模型 ======================
def train_model(df, predict_steps=8):
    """
    训练模型
    :param df: 数据DataFrame
    :param predict_steps: 预测步长（小时），默认8小时
    :return: 模型信息字典
    """
    # ======================================
    print("开始训练模型...")
    print(f"原始数据条数：{len(df)}")
    print(f"预测步长：{predict_steps} 小时")
    # ======================================

    df = clean_data(df)

    # ======================================
    print(f"数据清洗完成，剩余：{len(df)} 条")
    # ======================================

    time_diff = df["timestamp"].diff().dropna()
    sample_interval_min = time_diff.dt.total_seconds().median() / 60

    # ======================================
    print(f"采样间隔：{sample_interval_min:.2f} 分钟")
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
    print(f"特征工程完成，特征数据量：{df_feat.shape}")
    # ======================================

    feat_cols = [c for c in df_feat.columns if c not in ["timestamp", "id"] + list(target_map.values())]
    print(df_feat)
    n = len(df_feat)
    train = int(0.7 * n)
    val = int(0.15 * n)
    df_train = df_feat.iloc[:train]
    df_val = df_feat.iloc[train:train + val]
    df_test = df_feat.iloc[train + val:]

    # ======================================
    print(f"训练集：{len(df_train)} 条 | 验证集：{len(df_val)} 条 | 测试集：{len(df_test)} 条")
    # ======================================

    scaler = RobustScaler()
    model_dict = {}
    selected_feat = {}
    metrics_dict = {}

    for target_col, target_name in target_map.items():
        # ======================================
        print(f"\n开始训练目标：{target_col}")
        # ======================================

        y_train = df_train[target_name].values
        X_all = df_train[feat_cols].values
        X_all_s = scaler.fit_transform(X_all)

        selector = SelectFromModel(CatBoostRegressor(random_state=SEED, verbose=0))
        selector.fit(X_all_s, y_train)
        selected = [feat_cols[i] for i in range(len(feat_cols)) if selector.get_support()[i]]

        # ======================================
        print(f"特征选择完成，选中 {len(selected)} 个特征")
        # ======================================

        X_train = df_train[selected].values
        X_val = df_val[selected].values
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        tscv = TimeSeriesSplit(3)
        grid = GridSearchCV(
            CatBoostRegressor(random_state=SEED, verbose=0),
            {"max_depth": [3, 4, 5], "learning_rate": [0.04, 0.06], "n_estimators": [200, 300]},
            cv=tscv
        )
        grid.fit(X_train_s, y_train)
        model = grid.best_estimator_
        model.fit(X_train_s, y_train, eval_set=[(X_val_s, df_val[target_name].values)])

        model_dict[target_col] = model
        selected_feat[target_col] = selected

        # 在测试集上评估模型
        y_test = df_test[target_name].values
        X_test = df_test[selected].values
        X_test_s = scaler.transform(X_test)
        y_pred = model.predict(X_test_s)

        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        metrics_dict[target_col] = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }

        # ======================================
        print(f"{target_col} 训练完成！")
        print(f"测试集指标 - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        # ======================================

    # 最后保存模型，返回路径
    model_name = f"model_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"
    import os

    # 创建以模型名字命名的文件夹
    model_dir = os.path.join("models", model_name)
    os.makedirs(model_dir, exist_ok=True)

    scaler_path = os.path.join(model_dir, "scaler.pkl")
    model_path_file = os.path.join(model_dir, "models.pkl")
    config_path = os.path.join(model_dir, "config.pkl")

    joblib.dump(scaler, scaler_path)
    joblib.dump(model_dict, model_path_file)
    joblib.dump({
        "sample_interval_min": sample_interval_min,
        "selected_feat": selected_feat,
        "target_map": target_map,
        "metrics": metrics_dict
    }, config_path)

    # ======================================
    print(f"\n所有模型训练完成！已保存到文件夹: {model_dir}")
    print("scaler.pkl")
    print("models.pkl")
    print("config.pkl\n")
    # ======================================
    return {
        "model_name": model_name,
        "model_path": model_dir,
        "scaler_path": scaler_path,
        "config_path": config_path,
        "metrics": metrics_dict
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