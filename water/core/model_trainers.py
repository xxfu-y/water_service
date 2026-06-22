"""
模型训练器模块
支持多种机器学习模型的训练和自动选择
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectFromModel
from catboost import CatBoostRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings

warnings.filterwarnings('ignore')

SEED = 42


class ModelType:
    """模型类型枚举"""
    LSTM = "lstm"
    XGBOOST = "xgboost"
    CATBOOST = "catboost"
    LIGHTGBM = "lightboost"
    RF = "rf"
    AUTO = "auto"
    
    @classmethod
    def get_all_types(cls):
        """获取所有非auto的模型类型"""
        return [cls.LSTM, cls.XGBOOST, cls.CATBOOST, cls.LIGHTGBM, cls.RF]
    
    @classmethod
    def is_valid(cls, model_type):
        """验证模型类型是否有效"""
        valid_types = [cls.LSTM, cls.XGBOOST, cls.CATBOOST, cls.LIGHTGBM, cls.RF, cls.AUTO]
        return model_type.lower() in valid_types


class LSTMRegressor(BaseEstimator, RegressorMixin):
    """LSTM回归器，兼容sklearn接口"""
    
    def __init__(self, hidden_size=64, num_layers=2, 
                 learning_rate=0.001, epochs=100, batch_size=32, 
                 dropout=0.2, random_state=42, verbose=0):
        # 移除input_size参数，将在fit时自动推断
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.random_state = random_state
        self.verbose = verbose
        self.model = None
        self.device = None
        self.input_size = None  # 将在fit时设置
        
    def _build_model(self):
        """构建LSTM网络"""
        class LSTMNet(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout):
                super(LSTMNet, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0
                )
                self.fc = nn.Linear(hidden_size, 1)
                
            def forward(self, x):
                # x shape: (batch, seq_len, features)
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
                
                out, _ = self.lstm(x, (h0, c0))
                out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
                return out
        
        return LSTMNet(self.input_size, self.hidden_size, self.num_layers, self.dropout)
    
    def _create_sequences(self, X, y=None, seq_length=10):
        """将数据转换为序列格式"""
        X_seq = []
        y_seq = []
        
        # 确保序列长度不超过数据长度
        actual_seq_length = min(seq_length, len(X))
        if actual_seq_length < 2:
            actual_seq_length = 2
        
        for i in range(len(X) - actual_seq_length + 1):
            X_seq.append(X[i:i + actual_seq_length])
            if y is not None:
                y_seq.append(y[i + actual_seq_length - 1])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq) if y is not None else None
        
        return X_seq, y_seq
    
    def fit(self, X, y, **kwargs):
        """训练模型"""
        # 设置随机种子
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        
        # 自动推断input_size
        if self.input_size is None:
            self.input_size = X.shape[1] if X.ndim > 1 else 1
        
        # 确定设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 转换数据为序列格式
        seq_length = min(10, len(X) // 5)  # 动态设置序列长度
        if seq_length < 2:
            seq_length = 2
        
        X_seq, y_seq = self._create_sequences(X, y, seq_length)
        
        # 转换为张量
        X_tensor = torch.FloatTensor(X_seq).unsqueeze(-1) if X_seq.ndim == 2 else torch.FloatTensor(X_seq)
        y_tensor = torch.FloatTensor(y_seq).unsqueeze(-1)
        
        # 创建数据加载器
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # 构建模型（此时input_size已设置）
        self.model = self._build_model().to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # 训练
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if self.verbose > 0 and (epoch + 1) % 10 == 0:
                print(f"  Epoch [{epoch+1}/{self.epochs}], Loss: {total_loss/len(dataloader):.6f}")
        
        return self
    
    def predict(self, X):
        """预测"""
        if self.model is None:
            raise RuntimeError("模型尚未训练")
        
        self.model.eval()
        
        # 转换数据为序列格式
        seq_length = min(10, len(X) // 5)
        if seq_length < 2:
            seq_length = 2
        
        X_seq, _ = self._create_sequences(X, None, seq_length)
        
        # 转换为张量
        X_tensor = torch.FloatTensor(X_seq).unsqueeze(-1) if X_seq.ndim == 2 else torch.FloatTensor(X_seq)
        X_tensor = X_tensor.to(self.device)
        
        # 预测
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        return predictions.cpu().numpy().flatten()


def create_model(model_type, params=None):
    """
    根据模型类型创建模型实例
    
    Args:
        model_type: 模型类型字符串
        params: 模型参数字典（可选）
        
    Returns:
        模型实例
    """
    model_type = model_type.lower()
    
    if model_type == ModelType.LSTM:
        default_params = {
            'hidden_size': 64,
            'num_layers': 2,
            'learning_rate': 0.001,
            'epochs': 80,
            'batch_size': 32,
            'dropout': 0.2,
            'random_state': SEED,
            'verbose': 0
        }
        if params:
            default_params.update(params)
        return LSTMRegressor(**default_params)
    
    elif model_type == ModelType.XGBOOST:
        default_params = {
            'n_estimators': 400,
            'max_depth': 5,
            'learning_rate': 0.05,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': SEED,
            'verbose': 0,
            'objective': 'reg:squarederror',
            'n_jobs': -1
            # 注意：early_stopping_rounds不在这里设置，因为GridSearchCV不支持
        }
        if params:
            default_params.update(params)
        return xgb.XGBRegressor(**default_params)
    
    elif model_type == ModelType.CATBOOST:
        default_params = {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.05,
            'l2_leaf_reg': 3,
            'border_count': 128,
            'random_state': SEED,
            'verbose': 0,
            'thread_count': -1  # 使用所有CPU核心
            # 注意：early_stopping_rounds在fit时传入，不在这里设置
        }
        if params:
            default_params.update(params)
        return CatBoostRegressor(**default_params)
    
    elif model_type == ModelType.LIGHTGBM:
        default_params = {
            'n_estimators': 400,
            'max_depth': 5,
            'learning_rate': 0.05,
            'num_leaves': 50,
            'min_child_samples': 20,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': SEED,
            'verbose': -1,
            'force_col_wise': True,
            'n_jobs': -1
            # 注意：early_stopping_round不在这里设置，因为GridSearchCV不支持
        }
        if params:
            default_params.update(params)
        return lgb.LGBMRegressor(**default_params)
    
    elif model_type == ModelType.RF:
        default_params = {
            'n_estimators': 400,
            'max_depth': 12,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'random_state': SEED,
            'n_jobs': -1
        }
        if params:
            default_params.update(params)
        return RandomForestRegressor(**default_params)
    
    else:
        print(f"不支持的模型类型: {model_type}")
        raise ValueError(f"不支持的模型类型: {model_type}")


def get_param_grid(model_type):
    """
    获取模型超参数搜索网格（优化版：平衡速度和质量）
    
    Args:
        model_type: 模型类型字符串
        
    Returns:
        参数字典
    """
    model_type = model_type.lower()
    
    if model_type == ModelType.LSTM:
        return {
            'hidden_size': [64, 128],
            'num_layers': [2, 3],
            'learning_rate': [0.001, 0.005],
            'epochs': [60, 80]
        }
    elif model_type == ModelType.XGBOOST:
        return {
            'max_depth': [5, 6],
            'learning_rate': [0.05, 0.1],
            'n_estimators': [400],
            'subsample': [0.9],
            'colsample_bytree': [0.9]
        }
    elif model_type == ModelType.CATBOOST:
        return {
            'max_depth': [5, 6, 7],
            'learning_rate': [0.03, 0.05, 0.1],
            'n_estimators': [400, 500],
            'l2_leaf_reg': [3, 5],
            'border_count': [128]
        }
    elif model_type == ModelType.LIGHTGBM:
        return {
            'max_depth': [5, 6],
            'learning_rate': [0.05, 0.1],
            'n_estimators': [400],
            'num_leaves': [50],
            'min_child_samples': [20]
        }
    elif model_type == ModelType.RF:
        return {
            'max_depth': [10, 12],
            'n_estimators': [400],
            'min_samples_split': [5],
            'min_samples_leaf': [2]
        }
    else:
        print(f"不支持的模型类型: {model_type}")
        raise ValueError(f"不支持的模型类型: {model_type}")


def train_single_model(X_train_s, y_train, X_val_s, y_val, model_type, param_grid=None):
    """
    训练单个模型（优化版：支持早停和并行）
    
    Args:
        X_train_s: 标准化后的训练特征
        y_train: 训练标签
        X_val_s: 标准化后的验证特征
        y_val: 验证标签
        model_type: 模型类型
        param_grid: 参数网格（可选，默认使用预定义网格）
        
    Returns:
        包含模型和最佳参数的字典 {'model': model, 'best_params': params}
    """
    if param_grid is None:
        param_grid = get_param_grid(model_type)
    
    base_model = create_model(model_type)
    tscv = TimeSeriesSplit(3)  # 使用3折交叉验证（平衡速度和质量）
    
    grid = GridSearchCV(
        base_model,
        param_grid,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,  # 使用所有CPU核心
        refit=True,
        verbose=0,  # 关闭详细输出加快速度
        error_score='raise'  # 出现错误时立即抛出
    )
    
    try:
        grid.fit(X_train_s, y_train)
        best_model = grid.best_estimator_
        best_params = grid.best_params_
    except Exception as e:
        print(f"  ⚠️ {model_type} GridSearch失败: {e}")
        # 如果GridSearch失败，使用默认参数训练
        best_model = create_model(model_type)
        best_model.fit(X_train_s, y_train)
        best_params = {}  # 空字典表示使用默认参数
    
    # 对于支持早停的模型，在GridSearch后用eval_set进行额外训练
    # 注意：CatBoost/XGBoost/LightGBM的early_stopping需要在fit时传入，不能在初始化时设置
    if model_type.lower() == ModelType.CATBOOST:
        try:
            # CatBoost可以在fit时使用eval_set和early_stopping
            best_model.fit(
                X_train_s, y_train,
                eval_set=[(X_val_s, y_val)],
                verbose=0,
                early_stopping_rounds=50
            )
        except Exception as e:
            print(f"  ⚠️ CatBoost额外训练失败: {e}，继续使用GridSearch结果")
    elif model_type.lower() == ModelType.XGBOOST:
        try:
            # XGBoost需要在fit时传入eval_set来实现早停
            best_model.fit(
                X_train_s, y_train,
                eval_set=[(X_val_s, y_val)],
                verbose=0
            )
        except Exception as e:
            print(f"  ⚠️ XGBoost额外训练失败: {e}，继续使用GridSearch结果")
    elif model_type.lower() == ModelType.LIGHTGBM:
        try:
            # LightGBM需要在fit时传入eval_set来实现早停
            best_model.fit(
                X_train_s, y_train,
                eval_set=[(X_val_s, y_val)],
                verbose=-1
            )
        except Exception as e:
            print(f"  ⚠️ LightGBM额外训练失败: {e}，继续使用GridSearch结果")
    
    return {
        'model': best_model,
        'best_params': best_params
    }


def evaluate_model(model, X_test_s, y_test):
    """
    评估模型性能
    
    Args:
        model: 训练好的模型
        X_test_s: 标准化后的测试特征
        y_test: 测试标签
        
    Returns:
        包含RMSE、MAE、R²的字典
    """
    y_pred = model.predict(X_test_s)
    
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }


def train_with_model_type(X_train, y_train, X_val, y_val, X_test, y_test, 
                          model_type="catboost", scaler=None):
    """
    根据模型类型训练模型
    
    Args:
        X_train: 训练特征
        y_train: 训练标签
        X_val: 验证特征
        y_val: 验证标签
        X_test: 测试特征
        y_test: 测试标签
        model_type: 模型类型（lstm/xgboost/catboost/lightgbm/rf/auto）
        scaler: 标准化器（可选，如果为None则创建新的）
        
    Returns:
        包含模型、指标和选中特征的字典
    """
    model_type = model_type.lower()
    
    if not ModelType.is_valid(model_type):
        print(f"不支持的模型类型: {model_type}")
        raise ValueError(f"无效的模型类型: {model_type}")
    
    # 初始化或复用scaler
    if scaler is None:
        scaler = RobustScaler()
    
    # 特征选择
    selector = SelectFromModel(CatBoostRegressor(random_state=SEED, verbose=0))
    selector.fit(scaler.fit_transform(X_train), y_train)
    selected_mask = selector.get_support()
    selected_indices = [i for i, mask in enumerate(selected_mask) if mask]
    
    # 应用特征选择
    X_train_selected = X_train[:, selected_indices]
    X_val_selected = X_val[:, selected_indices]
    X_test_selected = X_test[:, selected_indices]
    
    # 标准化
    X_train_s = scaler.fit_transform(X_train_selected)
    X_val_s = scaler.transform(X_val_selected)
    X_test_s = scaler.transform(X_test_selected)
    
    result = {}
    
    # Auto模式：训练所有模型并选择最佳
    if model_type == ModelType.AUTO:
        print("Auto模式：训练所有模型并选择最佳...")
        best_rmse = float('inf')
        best_model = None
        best_metrics = None
        best_model_type = None
        best_params = None
        
        for mt in ModelType.get_all_types():
            try:
                print(f"  正在训练 {mt}...")
                result = train_single_model(X_train_s, y_train, X_val_s, y_val, mt)
                model = result['model']
                params = result['best_params']
                metrics = evaluate_model(model, X_test_s, y_test)
                
                print(f"    {mt} - RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, R2: {metrics['r2']:.4f}")
                
                if metrics['rmse'] < best_rmse:
                    best_rmse = metrics['rmse']
                    best_model = model
                    best_metrics = metrics
                    best_model_type = mt
                    best_params = params
                    
            except Exception as e:
                print(f"  警告: {mt} 训练失败: {str(e)}")
                continue
        
        if best_model is None:
            raise RuntimeError("Auto模式下所有模型训练均失败")
        
        print(f"\n✅ 最佳模型: {best_model_type}")
        print(f"   性能指标 - RMSE: {best_rmse:.4f}, MAE: {best_metrics['mae']:.4f}, R²: {best_metrics['r2']:.4f}")
        if best_params:
            print(f"   最优参数:")
            for param_name, param_value in best_params.items():
                print(f"     - {param_name}: {param_value}")
        else:
            print(f"   使用默认参数")
        print()
        
        result['model'] = best_model
        result['metrics'] = best_metrics
        result['model_type'] = best_model_type
        result['best_params'] = best_params
        
    else:
        # 指定模型类型训练
        print(f"正在训练 {model_type}...")
        result = train_single_model(X_train_s, y_train, X_val_s, y_val, model_type)
        model = result['model']
        params = result['best_params']
        metrics = evaluate_model(model, X_test_s, y_test)
        
        print(f"  ✅ {model_type} - RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, R2: {metrics['r2']:.4f}")
        if params:
            print(f"     最优参数:")
            for param_name, param_value in params.items():
                print(f"       - {param_name}: {param_value}")
        print()
        
        result['metrics'] = metrics
        result['model_type'] = model_type
    
    result['scaler'] = scaler
    result['selected_indices'] = selected_indices
    result['selected_mask'] = selected_mask
    
    return result

