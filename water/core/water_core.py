import pandas as pd
import numpy as np
import joblib
import sys
import os
from sklearn.preprocessing import RobustScaler

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from water.core.model_trainers import train_with_model_type, ModelType

SEED = 42


def hours_to_steps(hours, sample_interval_min):
    return int(round(hours * 60 / sample_interval_min))


# ====================== 特征工程 ======================
def build_features(df_input, sample_interval_min):
    """
    自适应构建特征 - 自动识别数据类型并生成相应特征
    
    Args:
        df_input: 输入数据DataFrame
        sample_interval_min: 采样间隔（分钟）
    
    Returns:
        包含构建特征的DataFrame
    """
    df = df_input.copy()
    dff = df.copy()
    
    # 自动识别列类型
    inflow_cols = [col for col in df.columns if 'inflow' in col.lower()]
    outflow_cols = [col for col in df.columns if 'outflow' in col.lower()]
    aeration_cols = [col for col in df.columns if any(kw in col.lower() for kw in ['aeration', 'blower'])]
    do_cols = [col for col in df.columns if 'do' in col.lower()]
    mlss_cols = [col for col in df.columns if 'mlss' in col.lower()]
    
    # 1. 进水变量滞后特征（历史值影响未来）
    for col in inflow_cols:
        if col not in df.columns:
            continue
        
        # 不同时间尺度的滞后（减少到关键时间点）
        for h in [0.5, 1, 2, 4]:
            s = hours_to_steps(h, sample_interval_min)
            dff[f"{col}_lag{h}h"] = df[col].shift(s)
        
        # 滚动均值（长周期趋势）
        for h in [8, 12, 24]:
            s = hours_to_steps(h, sample_interval_min)
            dff[f"{col}_mean_{h}h"] = df[col].shift(1).rolling(s).mean()
        
        # 指数加权移动平均（EWMA）- 只保留关键的
        for span in [12, 24]:
            steps = hours_to_steps(span, sample_interval_min)
            dff[f"{col}_ewma_{span}h"] = df[col].shift(1).ewm(span=steps, adjust=False).mean()
        
        # 差分特征（变化率）- 只保留1h和2h
        for h in [1, 2]:
            s = hours_to_steps(h, sample_interval_min)
            dff[f"{col}_diff{h}h"] = df[col].diff(s)
    
    # 2. 出水变量滞后特征（包括目标变量的历史值）
    for col in outflow_cols:
        if col not in df.columns:
            continue
        
        # 减少滞后的时间点
        for h in [1, 2, 4, 8, 12]:
            s = hours_to_steps(h, sample_interval_min)
            dff[f"{col}_lag{h}h"] = df[col].shift(s)
    
    # 3. 曝气/鼓风机变量
    if aeration_cols:
        available_aeration = [col for col in aeration_cols if col in df.columns]
        if available_aeration:
            dff[available_aeration] = df[available_aeration]
            
            # 曝气总量
            numeric_aeration = df[available_aeration].select_dtypes(include=[np.number])
            if not numeric_aeration.empty:
                dff["MV_total"] = numeric_aeration.sum(axis=1)
                
                # MV 滞后（减少时间点）
                for h in [0.5, 1, 2]:
                    s = hours_to_steps(h, sample_interval_min)
                    dff[f"MV_total_lag{h}h"] = dff["MV_total"].shift(s)
                
                # 长周期累积
                dff["aeration_sum_8h"] = dff["MV_total"].shift(1).rolling(hours_to_steps(8, sample_interval_min)).sum()
                dff["aeration_sum_24h"] = dff["MV_total"].shift(1).rolling(hours_to_steps(24, sample_interval_min)).sum()
    
    # 4. 溶解氧(DO)特征
    for col in do_cols:
        if col not in df.columns:
            continue
        
        # DO 滞后（减少时间点）
        for h in [0.5, 1, 2, 3]:
            s = hours_to_steps(h, sample_interval_min)
            dff[f"{col}_lag{h}h"] = df[col].shift(s)
        
        # DO 滚动均值（只保留关键的）
        for h in [3, 6]:
            s = hours_to_steps(h, sample_interval_min)
            dff[f"{col}_roll_mean_{h}h"] = df[col].shift(1).rolling(s).mean()
    
    # 5. MLSS特征
    for col in mlss_cols:
        if col not in df.columns:
            continue
        
        dff[col] = df[col]
        
        # MLSS 滚动均值（只保留24h）
        s = hours_to_steps(24, sample_interval_min)
        dff[f"{col}_mean_24h"] = df[col].shift(1).rolling(s).mean()
    
    # 6. 交互特征（进水与出水的比值、差值等）
    for inflow_col in inflow_cols[:2]:  # 限制数量避免过多特征
        for outflow_col in outflow_cols[:2]:
            if inflow_col in df.columns and outflow_col in df.columns:
                # 比值特征
                dff[f"{inflow_col}_to_{outflow_col}_ratio"] = df[inflow_col] / (df[outflow_col] + 1e-6)
                # 差值特征
                dff[f"{inflow_col}_minus_{outflow_col}"] = df[inflow_col] - df[outflow_col]
    
    return dff
    
    return dff


def clean_data(df):
    """
    数据清洗 - 使用更鲁棒的方法处理异常值和缺失值
    
    Args:
        df: 输入DataFrame
        
    Returns:
        清洗后的DataFrame
    """
    # 时间戳处理
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for c in numeric_cols:
        # 使用IQR方法检测异常值（比3σ更鲁棒）
        Q1 = df[c].quantile(0.25)
        Q3 = df[c].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 2.5 * IQR
        upper_bound = Q3 + 2.5 * IQR
        
        # 裁剪异常值
        df[c] = df[c].clip(lower_bound, upper_bound)
        
        # 使用线性插值填补缺失值
        df[c] = df[c].interpolate(method='linear', limit=10).ffill().bfill()
    
    return df


# ====================== 训练模型 ======================
def train_model(df, predict_steps=8, model_type="catboost", model_name=None, 
                target_variable=None, sampling_interval=300):
    """
    训练模型
    
    Args:
        df: 数据DataFrame
        predict_steps: 预测步数（采样点数量），
        model_type: 模型类型（lstm/xgboost/catboost/lightgbm/rf/auto），默认catboost
        model_name: 模型名称，如果未提供则使用时间戳生成
        target_variable: 需要预测的目标变量名或列表
        sampling_interval: 采样间隔（秒），默认300秒
    
    Returns:
        模型信息字典
    """
    print("开始训练模型...")
    print(f"原始数据条数: {len(df)}")
    print(f"预测步数: {predict_steps} steps")
    print(f"模型类型: {model_type}")
    print(f"模型名称: {model_name if model_name else '自动生成'}")
    print(f"目标变量: {target_variable}")
    print(f"采样间隔: {sampling_interval} seconds")

    df = clean_data(df)

    print(f"数据清洗完成, 剩余: {len(df)} records")

    # 计算采样间隔（分钟）
    time_diff = df["timestamp"].diff().dropna()
    sample_interval_min = time_diff.dt.total_seconds().median() / 60
    
    # 如果提供了sampling_interval参数，优先使用
    if sampling_interval and sampling_interval > 0:
        sample_interval_min = sampling_interval / 60.0

    print(f"采样间隔: {sample_interval_min:.2f} minutes")

    # 确定目标变量
    if target_variable is None or (isinstance(target_variable, list) and len(target_variable) == 0):
        # 自动识别目标变量（通常是outflow开头的列）
        target_variable = [col for col in df.columns if 'outflow' in col.lower()]
        if not target_variable:
            raise ValueError("无法自动识别目标变量，请明确指定target_variable参数")
    
    if isinstance(target_variable, str):
        target_variable = [target_variable]
    
    # 验证目标变量是否存在于数据中
    missing_targets = [t for t in target_variable if t not in df.columns]
    if missing_targets:
        raise ValueError(f"目标变量不存在于数据中: {missing_targets}")
    
    print(f"将预测以下目标变量: {target_variable}")

    # 构建特征
    df_feat = build_features(df, sample_interval_min)

    # 创建目标变量的超前标签（lead features）
    target_map = {}
    for target_col in target_variable:
        # predict_steps 直接就是步数，不需要转换
        target_name = f"{target_col}_lead{predict_steps}steps"
        target_map[target_col] = target_name
        df_feat[target_name] = df[target_col].shift(-predict_steps)

    # 删除含有NaN的行
    df_feat = df_feat.dropna().reset_index(drop=True)

    print(f"特征工程完成, 特征数据量: {df_feat.shape}")

    # 分离特征列和目标列
    exclude_cols = ["timestamp", "id"] + list(target_map.values())
    feat_cols = [c for c in df_feat.columns if c not in exclude_cols]
    
    print(f"特征列数量: {len(feat_cols)}")
    print(df_feat.head())
    
    # 划分训练集、验证集、测试集
    n = len(df_feat)
    train = int(0.7 * n)
    val = int(0.15 * n)
    df_train = df_feat.iloc[:train]
    df_val = df_feat.iloc[train:train + val]
    df_test = df_feat.iloc[train + val:]

    print(f"训练集: {len(df_train)} | 验证集: {len(df_val)} | 测试集: {len(df_test)}")

    model_dict = {}
    selected_feat = {}
    metrics_dict = {}
    scaler = None  # 复用scaler

    for target_col, target_name in target_map.items():
        print(f"\n开始训练目标: {target_col}")

        y_train = df_train[target_name].values
        X_train = df_train[feat_cols].values
        X_val = df_val[feat_cols].values
        X_test = df_test[feat_cols].values

        # 使用新的模型训练模块
        result = train_with_model_type(
            X_train, y_train, 
            X_val, df_val[target_name].values,
            X_test, df_test[target_name].values,
            model_type=model_type,
            scaler=scaler
        )
        
        # 复用scaler以保持一致性
        if scaler is None:
            scaler = result['scaler']
        
        model_dict[target_col] = result['model']
        selected_feat[target_col] = [
            feat_cols[i] for i in result['selected_indices']
        ]
        metrics_dict[target_col] = result['metrics']

        print(f"{target_col} 训练完成!")
        print(f"使用模型: {result['model_type']}")
        print(f"测试集指标 - RMSE: {result['metrics']['rmse']:.4f}, MAE: {result['metrics']['mae']:.4f}, R2: {result['metrics']['r2']:.4f}")

    # 保存模型
    if model_name is None or model_name.strip() == "":
        model_name = f"model_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"
    
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
        "predict_steps": predict_steps,
        "target_variable": target_variable,
        "selected_feat": selected_feat,
        "target_map": target_map,
        "metrics": metrics_dict
    }, config_path)

    print(f"\n所有模型训练完成! 已保存到文件夹: {model_dir}")
    print("scaler.pkl")
    print("models.pkl")
    print("config.pkl\n")
    
    return {
        "model_name": model_name,
        "model_path": model_dir,
        "scaler_path": scaler_path,
        "config_path": config_path,
        "metrics": metrics_dict
    }


# ====================== 预测 ======================
def predict_model(df, model_path=None):
    """
    使用训练好的模型进行预测
    
    Args:
        df: 输入数据DataFrame
        model_path: 模型路径，如果为None则使用默认路径
    
    Returns:
        预测结果字典
    """
    if model_path is None:
        scaler = joblib.load("scaler.pkl")
        model_dict = joblib.load("models.pkl")
        cfg = joblib.load("model_config.pkl")
    else:
        scaler = joblib.load(os.path.join(model_path, "scaler.pkl"))
        model_dict = joblib.load(os.path.join(model_path, "models.pkl"))
        cfg = joblib.load(os.path.join(model_path, "config.pkl"))

    sample_interval_min = cfg["sample_interval_min"]
    target_variable = cfg.get("target_variable", [])
    selected_feat = cfg["selected_feat"]
    
    df = clean_data(df)
    # 暂时不做特征工程，直接使用原始数据
    df_feat = df.copy()
    df_feat = df_feat.ffill().bfill()

    result = {}
    for target_col in target_variable:
        feats = selected_feat[target_col]
        X = df_feat[feats].values
        X_s = scaler.transform(X)
        pred = model_dict[target_col].predict(X_s)
        result[target_col] = float(pred[-1])
    
    return result