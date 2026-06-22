import pandas as pd
import numpy as np
import joblib
import warnings
from concurrent import futures
import grpc
from catboost import CatBoostRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

# 自动生成的 gRPC 代码
import water.proto.water_pb2 as water_pb2
import water.proto.water_pb2_grpc as water_pb2_grpc
from water.utils.config import get_grpc_address, get_max_workers

# ====================== 全局配置（不变） ======================
SEED = 42
np.random.seed(SEED)
warnings.filterwarnings("ignore")

DV_CONFIG = {
    "inflow_cod": -4.5,
    "inflow_nh3": -4.5,
    "inflow_rate3": -0.05,
    "bio3_mlss_meas": 0.0
}

MV_COLS = [
    "bio3_front_aeration_flow_meas",
    "bio3_end_aeration_flow_meas"
]

CV_CONFIG = {
    "bio3_end_do_meas": 0.5,
    "outflow_cod": 8.0,
    "outflow_nh3": 8.0
}

# ====================== 工具函数 ======================
def hours_to_steps(hours, sample_interval_min):
    return int(round(hours * 60 / sample_interval_min))

# ====================== 特征工程（从传入的 df 计算） ======================
def build_features(df_input, sample_interval_min):
    df = df_input.copy()
    dff = df.copy()

    # 1. DV 滞后
    for col, offset in DV_CONFIG.items():
        steps = hours_to_steps(abs(offset), sample_interval_min)
        dff[f"{col}_lag{steps}"] = df[col].shift(steps)

    # 2. MV
    dff[MV_COLS] = df[MV_COLS]
    dff["MV_total"] = df[MV_COLS].sum(axis=1)

    # 3. 当前状态
    dff[list(CV_CONFIG.keys())] = df[list(CV_CONFIG.keys())]

    # 4. DO 滞后
    for h in [0.5, 1, 2, 3]:
        s = hours_to_steps(h, sample_interval_min)
        dff[f"bio3_end_do_meas_lag{h}h"] = df["bio3_end_do_meas"].shift(s)

    # 5. MV 滞后
    for h in [0.5, 1, 2]:
        s = hours_to_steps(h, sample_interval_min)
        dff[f"MV_total_lag{h}h"] = dff["MV_total"].shift(s)

    # 6. DO 滚动
    for h in [1, 3, 6]:
        s = hours_to_steps(h, sample_interval_min)
        dff[f"do_roll_mean_{h}h"] = df["bio3_end_do_meas"].shift(1).rolling(s).mean()

    # 7. 进水滚动
    for h in [4, 8, 12, 24]:
        s = hours_to_steps(h, sample_interval_min)
        dff[f"inflow_cod_mean_{h}h"] = df["inflow_cod"].shift(1).rolling(s).mean()
        dff[f"inflow_nh3_mean_{h}h"] = df["inflow_nh3"].shift(1).rolling(s).mean()

    # 8. 长周期
    dff["aeration_sum_8h"] = dff["MV_total"].shift(1).rolling(hours_to_steps(8, sample_interval_min)).sum()
    dff["aeration_sum_24h"] = dff["MV_total"].shift(1).rolling(hours_to_steps(24, sample_interval_min)).sum()
    dff["mlss_mean_8h"] = df["bio3_mlss_meas"].shift(1).rolling(hours_to_steps(8, sample_interval_min)).mean()
    dff["mlss_mean_24h"] = df["bio3_mlss_meas"].shift(1).rolling(hours_to_steps(24, sample_interval_min)).mean()

    # 9. 出水滞后
    for h in [2, 4, 8, 12, 24]:
        s = hours_to_steps(h, sample_interval_min)
        dff[f"outflow_cod_lag{h}h"] = df["outflow_cod"].shift(s)
        dff[f"outflow_nh3_lag{h}h"] = df["outflow_nh3"].shift(s)

    return dff

# ====================== 训练 + 预测（从 df 训练，返回最后一条预测） ======================
def train_and_predict(df: pd.DataFrame):
    # 时间处理
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    # 自动计算采样间隔
    time_diff = df["timestamp"].diff().dropna()
    sample_interval_min = time_diff.dt.total_seconds().median() / 60

    # 异常值 + 缺失值
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        m = df[col].mean()
        s = df[col].std()
        if s > 0:
            df[col] = df[col].clip(m - 3*s, m + 3*s)
    df[numeric_cols] = df[numeric_cols].interpolate(limit=10).ffill().bfill()

    # 特征工程
    df_feat = build_features(df, sample_interval_min)

    # 构建标签
    target_map = {}
    for col, lead in CV_CONFIG.items():
        steps = hours_to_steps(lead, sample_interval_min)
        target = f"{col}_lead{lead}h"
        target_map[col] = target
        df_feat[target] = df[col].shift(-steps)

    df_feat = df_feat.dropna().reset_index(drop=True)
    feat_cols = [c for c in df_feat.columns if c not in ["timestamp"] + list(target_map.values())]

    # 划分
    n = len(df_feat)
    train = int(0.7 * n)
    val = int(0.15 * n)
    df_train = df_feat.iloc[:train]
    df_val = df_feat.iloc[train:train+val]
    df_test = df_feat.iloc[train+val:]

    # 训练
    scaler = RobustScaler()
    model_dict = {}
    selected_feat = {}

    for target_col, target_name in target_map.items():
        y_train = df_train[target_name].values
        X_all = df_train[feat_cols].values
        X_all_s = scaler.fit_transform(X_all)

        selector = SelectFromModel(CatBoostRegressor(random_state=SEED, verbose=0))
        selector.fit(X_all_s, y_train)
        selected = [feat_cols[i] for i in range(len(feat_cols)) if selector.get_support()[i]]

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

    # 预测最后一条
    result = {}
    for target_col in CV_CONFIG.keys():
        feats = selected_feat[target_col]
        X = df_feat[feats].values
        X_s = scaler.transform(X)
        pred = model_dict[target_col].predict(X_s)
        result[target_col] = float(pred[-1])

    return result

# ====================== gRPC 服务 ======================
class WaterService(water_pb2_grpc.WaterServiceServicer):
    def TrainAndPredict(self, request, context):
        # 从 gRPC 转成 DataFrame（**不读任何本地文件！**）
        rows = []
        for row in request.rows:
            rows.append({
                "timestamp": row.timestamp,
                "inflow_cod": row.inflow_cod,
                "inflow_nh3": row.inflow_nh3,
                "inflow_rate3": row.inflow_rate3,
                "bio3_mlss_meas": row.bio3_mlss_meas,
                "bio3_front_aeration_flow_meas": row.bio3_front_aeration_flow_meas,
                "bio3_end_aeration_flow_meas": row.bio3_end_aeration_flow_meas,
                "bio3_end_do_meas": row.bio3_end_do_meas,
                "outflow_cod": row.outflow_cod,
                "outflow_nh3": row.outflow_nh3
            })

        df = pd.DataFrame(rows)
        result = train_and_predict(df)

        return water_pb2.TrainAndPredictResponse(
            bio3_end_do_meas=result["bio3_end_do_meas"],
            outflow_cod=result["outflow_cod"],
            outflow_nh3=result["outflow_nh3"]
        )

# ====================== 启动 ======================
def serve():
    max_workers = get_max_workers("train_service")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    water_pb2_grpc.add_WaterServiceServicer_to_server(WaterService(), server)
    address = get_grpc_address("train_service")
    server.add_insecure_port(address)
    server.start()
    print(f"gRPC 服务已启动：{address}")
    print("数据全部来自 gRPC，不读取任何本地 CSV！")
    import time
    while True:
        time.sleep(3600)

if __name__ == "__main__":
    serve()