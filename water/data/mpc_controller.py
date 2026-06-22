"""
MPC 控制器管理器 - 支持多实例和双模型运行
边缘服务器上的 MPC 控制核心
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import threading

from water.data.data_adapter import DataAdapter
from water.data.mpc_database import MpcDatabase


class InstanceModelManager:
    """实例模型管理器 - 管理正式模型和影子模型"""
    
    def __init__(self, instance_id: str):
        self.instance_id = instance_id
        self.formal_model = None  # 正式模型
        self.shadow_model = None  # 影子模型
        self.formal_model_path = None
        self.shadow_model_path = None
        self.model_lock = threading.Lock()
        
        print(f"[InstanceModelManager] 实例 {instance_id} 模型管理器初始化")
    
    def load_formal_model(self, model_path: str) -> bool:
        """加载正式模型"""
        try:
            with self.model_lock:
                model_data = self._load_model_from_path(model_path)
                if model_data:
                    self.formal_model = model_data
                    self.formal_model_path = model_path
                    print(f"[MPC] 实例 {self.instance_id} 正式模型加载成功: {model_path}")
                    return True
                return False
        except Exception as e:
            print(f"[MPC ERROR] 加载正式模型失败: {e}")
            return False
    
    def load_shadow_model(self, model_path: str) -> bool:
        """加载影子模型"""
        try:
            with self.model_lock:
                model_data = self._load_model_from_path(model_path)
                if model_data:
                    self.shadow_model = model_data
                    self.shadow_model_path = model_path
                    print(f"[MPC] 实例 {self.instance_id} 影子模型加载成功: {model_path}")
                    return True
                return False
        except Exception as e:
            print(f"[MPC ERROR] 加载影子模型失败: {e}")
            return False
    
    def _load_model_from_path(self, model_path: str) -> Optional[Dict]:
        """从路径加载模型"""
        try:
            model_dir = Path(model_path)
            
            # 加载模型、scaler 和配置
            models_file = model_dir / "models.pkl"
            scaler_file = model_dir / "scaler.pkl"
            config_file = model_dir / "config.pkl"
            
            if not all([models_file.exists(), scaler_file.exists()]):
                print(f"[MPC WARNING] 模型文件不完整: {model_path}")
                return None
            
            model = joblib.load(models_file)
            scaler = joblib.load(scaler_file)
            config = joblib.load(config_file) if config_file.exists() else {}
            
            return {
                "model": model,
                "scaler": scaler,
                "config": config,
                "loaded_at": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"[MPC ERROR] 模型加载错误: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def predict_with_formal(self, features: np.ndarray) -> Optional[np.ndarray]:
        """使用正式模型预测"""
        if self.formal_model is None:
            return None
        
        try:
            scaler = self.formal_model["scaler"]
            model = self.formal_model["model"]
            
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)
            
            return prediction
        except Exception as e:
            print(f"[MPC ERROR] 正式模型预测失败: {e}")
            return None
    
    def predict_with_shadow(self, features: np.ndarray) -> Optional[np.ndarray]:
        """使用影子模型预测"""
        if self.shadow_model is None:
            return None
        
        try:
            scaler = self.shadow_model["scaler"]
            model = self.shadow_model["model"]
            
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)
            
            return prediction
        except Exception as e:
            print(f"[MPC ERROR] 影子模型预测失败: {e}")
            return None
    
    def get_model_status(self) -> Dict:
        """获取模型状态"""
        return {
            "instance_id": self.instance_id,
            "has_formal_model": self.formal_model is not None,
            "has_shadow_model": self.shadow_model is not None,
            "formal_model_path": self.formal_model_path,
            "shadow_model_path": self.shadow_model_path,
            "formal_loaded_at": self.formal_model.get("loaded_at") if self.formal_model else None,
            "shadow_loaded_at": self.shadow_model.get("loaded_at") if self.shadow_model else None
        }
    
    def promote_shadow_to_formal(self) -> bool:
        """将影子模型提升为正式模型（自动切换）"""
        try:
            with self.model_lock:
                if self.shadow_model is None:
                    print(f"[MPC WARNING] 没有影子模型可提升")
                    return False
                
                # 影子模型变为正式模型
                self.formal_model = self.shadow_model
                self.formal_model_path = self.shadow_model_path
                
                print(f"[MPC] 实例 {self.instance_id} 影子模型已提升为正式模型")
                return True
        except Exception as e:
            print(f"[MPC ERROR] 模型提升失败: {e}")
            return False


class MpcController:
    """MPC 控制器类 - 支持实例级别的控制"""
    
    # 共享数据库实例
    _db_instance = None
    _db_lock = threading.Lock()
    
    @classmethod
    def get_database(cls, db_path: str = "mpc_data.db") -> MpcDatabase:
        """获取单例数据库实例"""
        if cls._db_instance is None:
            with cls._db_lock:
                if cls._db_instance is None:
                    cls._db_instance = MpcDatabase(db_path)
        return cls._db_instance
    
    def __init__(self, instance_id: str, config: Dict):
        """
        初始化 MPC 控制器
        
        Args:
            instance_id: 实例 ID
            config: MPC 控制器配置字典
        """
        self.instance_id = instance_id
        self.config = config
        self.controller_name = config.get("controllerName", f"mpc-{instance_id}")
        
        # 数据库实例
        self.db = self.get_database()
        
        # MPC 是否开启
        self.mpc_enabled = config.get("mpcEnabled", True)
        
        # 模型管理器
        self.model_manager = InstanceModelManager(instance_id)
        
        # 基本参数
        self.sampling_time = config.get("samplingTime", 300.0)
        self.prediction_horizon = config.get("predictionHorizon", 10)
        self.control_horizon = config.get("controlHorizon", 5)
        
        # 目标设定值
        self.target_sp = config.get("targetSP", 50.0)
        self.deadband_lower = config.get("deadbandLower", 49.5)
        self.deadband_upper = config.get("deadbandUpper", 50.5)
        
        # 安全限制
        self.safety_lower = config.get("safetyLower", 45.0)
        self.safety_upper = config.get("safetyUpper", 55.0)
        
        # MV 限制
        self.mv_physical_lower = config.get("mvPhysicalLower", 0.0)
        self.mv_physical_upper = config.get("mvPhysicalUpper", 100.0)
        self.mv_min_change = config.get("mvMinChange", 1.0)
        self.mv_max_change = config.get("mvMaxChange", 10.0)
        
        # 特征绑定
        self.feature_bindings = config.get("featureBindings", [])
        self.cv_tag = config.get("cvTag", "")
        self.mv_tag = config.get("mvTag", "")
        
        # 健康度阈值
        health_config = config.get("healthThresholdConfig", {})
        self.good_threshold = health_config.get("goodThreshold", 0.9)
        self.normal_threshold = health_config.get("normalThreshold", 0.7)
        
        # 自动切换配置
        self.auto_switch_config = config.get("autoSwitchConfig", {})
        self.auto_switch_enabled = config.get("autoSwitchToFormal", False)
        
        # 历史数据
        self.control_history = []
        self.performance_metrics = {
            "formal_rmse": [],
            "shadow_rmse": [],
            "switch_count": 0
        }
        
        print(f"[MPC] 控制器 {self.controller_name} (实例: {instance_id}) 初始化完成")
    
    def compute_control(self, sensor_data: List[Dict], 
                       use_shadow: bool = False) -> Dict:
        """
        计算最优控制量
        
        Args:
            sensor_data: 传感器数据列表
            use_shadow: 是否使用影子模型进行预测
            
        Returns:
            Dict: 控制结果
        """
        try:
            # 1. 获取当前 CV 值
            current_cv = self._get_current_cv(sensor_data)
            
            # 2. 检查是否在死区内
            if self.deadband_lower <= current_cv <= self.deadband_upper:
                return self._create_response(
                    mv_value=self._get_last_mv(),
                    predicted_cv=current_cv,
                    status="DEADBAND",
                    message="CV 在死区内，保持当前 MV",
                    model_used="NONE"
                )
            
            # 3. 选择预测模型
            if use_shadow:
                predict_func = self.model_manager.predict_with_shadow
                model_type = "SHADOW"
            else:
                predict_func = self.model_manager.predict_with_formal
                model_type = "FORMAL"
            
            # 4. 执行 MPC 优化
            if predict_func is None or (use_shadow and self.model_manager.shadow_model is None) or \
               (not use_shadow and self.model_manager.formal_model is None):
                # 如果没有可用模型，使用简化策略
                mv_value = self._simple_control(current_cv)
                predicted_cv = current_cv
                model_type = "SIMPLIFIED"
            else:
                # 使用 MPC 优化算法
                mv_value, predicted_cv = self._mpc_optimization(
                    sensor_data, 
                    predict_func
                )
            
            # 5. 应用 MV 约束
            mv_value = self._apply_mv_constraints(mv_value)
            
            # 6. 检查安全性
            if not self._check_safety(predicted_cv):
                return self._create_response(
                    mv_value=mv_value,
                    predicted_cv=predicted_cv,
                    status="ERROR",
                    message="预测值超出安全范围",
                    model_used=model_type
                )
            
            # 7. 计算健康度
            health_score = self._calculate_health_score(current_cv, predicted_cv)
            
            # 8. 记录控制和性能
            self._record_control(current_cv, mv_value, predicted_cv, model_type)
            
            # 9. 确定控制状态
            status = self._determine_status(current_cv, mv_value)
            
            # 10. 数据持久化（包含预测时间）
            self._save_to_database(
                current_cv=current_cv,
                mv_value=mv_value,
                predicted_cv=predicted_cv,
                health_score=health_score,
                model_type=model_type,
                status=status
            )
            
            # 10. 数据持久化
            self._save_to_database(
                current_cv=current_cv,
                mv_value=mv_value,
                predicted_cv=predicted_cv,
                health_score=health_score,
                model_type=model_type,
                status=status
            )
            
            return self._create_response(
                mv_value=mv_value,
                predicted_cv=predicted_cv,
                status=status,
                health_score=health_score,
                message="控制计算成功",
                model_used=model_type
            )
            
        except Exception as e:
            print(f"[MPC ERROR] 控制计算失败: {e}")
            import traceback
            traceback.print_exc()
            
            return self._create_response(
                mv_value=0.0,
                predicted_cv=0.0,
                status="ERROR",
                message=f"控制计算失败: {str(e)}",
                model_used="ERROR"
            )
    
    def _get_current_cv(self, sensor_data: List[Dict]) -> float:
        """获取当前 CV 值"""
        if not sensor_data:
            return self.target_sp
        
        latest_data = sensor_data[-1]
        values = latest_data.get("values", {})
        
        for binding in self.feature_bindings:
            if binding.get("type") == "CV":
                feature_name = binding.get("featureName")
                if feature_name in values:
                    return float(values[feature_name])
        
        return self.target_sp
    
    def _get_last_mv(self) -> float:
        """获取上一个 MV 值"""
        if self.control_history:
            return self.control_history[-1]["mv_value"]
        return self.target_sp
    
    def _simple_control(self, current_cv: float) -> float:
        """简化控制策略（PID 风格）"""
        error = self.target_sp - current_cv
        kp = 2.0
        mv_change = kp * error
        mv_change = np.clip(mv_change, -self.mv_max_change, self.mv_max_change)
        
        last_mv = self._get_last_mv()
        return last_mv + mv_change
    
    def _mpc_optimization(self, sensor_data: List[Dict], 
                         predict_func) -> Tuple[float, float]:
        """MPC 优化算法"""
        best_mv = self._get_last_mv()
        best_cost = float('inf')
        best_predicted_cv = self.target_sp
        
        # 在 MV 范围内搜索最优解
        mv_candidates = np.linspace(
            self.mv_physical_lower,
            self.mv_physical_upper,
            20
        )
        
        for mv_candidate in mv_candidates:
            try:
                # 构建特征并预测
                features = self._build_features(sensor_data, mv_candidate)
                predicted_trajectory = predict_func(features)
                
                if predicted_trajectory is None or len(predicted_trajectory) == 0:
                    continue
                
                # 计算代价函数
                cost = self._calculate_cost(predicted_trajectory, mv_candidate)
                
                if cost < best_cost:
                    best_cost = cost
                    best_mv = mv_candidate
                    best_predicted_cv = float(predicted_trajectory[-1])
                    
            except Exception as e:
                print(f"[MPC] 预测失败: {e}")
                continue
        
        return best_mv, best_predicted_cv
    
    def _build_features(self, sensor_data: List[Dict], mv_value: float) -> np.ndarray:
        """
        构建特征向量
        
        使用数据适配器统一处理特征提取
        
        Args:
            sensor_data: 传感器数据列表
            mv_value: MV 值
            
        Returns:
            特征数组
        """
        if not sensor_data:
            return np.array([[mv_value]])
        
        # 使用数据适配器提取特征
        adapter = DataAdapter()
        features_dict = adapter.extract_features_for_mpc(sensor_data, self.feature_bindings)
        
        # 如果没有绑定特征，使用默认方式
        if not features_dict:
            latest_data = sensor_data[-1]
            values = latest_data.get("values", {})
            
            # 提取所有数值特征
            features = []
            for key, value in values.items():
                try:
                    features.append(float(value))
                except (ValueError, TypeError):
                    continue
        else:
            # 使用绑定的特征
            features = list(features_dict.values())
        
        # 添加 MV 值
        features.append(mv_value)
        
        return np.array([features])
    
    def _calculate_cost(self, predicted_trajectory: List[float], 
                       mv_value: float) -> float:
        """计算代价函数"""
        if not predicted_trajectory:
            return float('inf')
        
        # 跟踪误差
        tracking_error = sum((y - self.target_sp) ** 2 for y in predicted_trajectory)
        
        # MV 变化惩罚
        last_mv = self._get_last_mv()
        mv_change_penalty = 0.1 * (mv_value - last_mv) ** 2
        
        return tracking_error + mv_change_penalty
    
    def _apply_mv_constraints(self, mv_value: float) -> float:
        """应用 MV 约束"""
        mv_value = np.clip(mv_value, self.mv_physical_lower, self.mv_physical_upper)
        
        last_mv = self._get_last_mv()
        mv_change = mv_value - last_mv
        mv_change = np.clip(mv_change, -self.mv_max_change, self.mv_max_change)
        
        return last_mv + mv_change
    
    def _check_safety(self, predicted_cv: float) -> bool:
        """检查安全性"""
        return self.safety_lower <= predicted_cv <= self.safety_upper
    
    def _calculate_health_score(self, current_cv: float, 
                               predicted_cv: float) -> float:
        """计算健康度评分"""
        error = abs(predicted_cv - self.target_sp)
        max_error = self.safety_upper - self.safety_lower
        
        if max_error == 0:
            return 1.0
        
        health_score = max(0.0, 1.0 - (error / max_error))
        return health_score
    
    def _record_control(self, cv_value: float, mv_value: float, 
                       predicted_cv: float, model_type: str):
        """记录控制历史"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "cv_value": cv_value,
            "mv_value": mv_value,
            "predicted_cv": predicted_cv,
            "model_type": model_type
        }
        self.control_history.append(record)
        
        if len(self.control_history) > 1000:
            self.control_history = self.control_history[-1000:]
    
    def _determine_status(self, current_cv: float, mv_value: float) -> str:
        """确定控制状态"""
        if mv_value >= self.mv_physical_upper * 0.95 or \
           mv_value <= self.mv_physical_lower * 1.05:
            return "SATURATED"
        
        if self.deadband_lower <= current_cv <= self.deadband_upper:
            return "DEADBAND"
        
        return "NORMAL"
    
    def _create_response(self, mv_value: float, predicted_cv: float,
                        status: str, message: str, 
                        health_score: float = 1.0,
                        model_used: str = "UNKNOWN") -> Dict:
        """创建响应字典"""
        return {
            "success": status != "ERROR",
            "message": message,
            "controller_name": self.controller_name,
            "instance_id": self.instance_id,
            "mv_value": float(mv_value),
            "predicted_cv": float(predicted_cv),
            "control_status": status,
            "health_score": float(health_score),
            "model_used": model_used,
            "timestamp": datetime.now().isoformat()
        }
    
    def update_config(self, new_config: Dict):
        """更新控制器配置"""
        self.config.update(new_config)
        print(f"[MPC] 实例 {self.instance_id} 配置已更新")
    
    def get_status(self) -> Dict:
        """获取控制器状态"""
        model_status = self.model_manager.get_model_status()
        
        return {
            "controller_name": self.controller_name,
            "instance_id": self.instance_id,
            "status": "RUNNING",
            "last_mv": self._get_last_mv(),
            "history_count": len(self.control_history),
            "models": model_status,
            "timestamp": datetime.now().isoformat()
        }
    
    def _save_to_database(self, current_cv: float, mv_value: float,
                         predicted_cv: float, health_score: float,
                         model_type: str, status: str):
        """
        保存数据到数据库
        
        逻辑：
        - 如果 MPC 开启：保存 MV、CV(target)、RMSE、预测值、预测时间
        - 如果 MPC 关闭：只保存 RMSE、预测值、预测时间
        
        Args:
            current_cv: 当前 CV 实际值
            mv_value: 计算出的 MV 值
            predicted_cv: 预测的 CV 值
            health_score: 健康度评分
            model_type: 使用的模型类型
            status: 控制状态
        """
        try:
            timestamp = datetime.now().isoformat()
            predict_time = timestamp  # 预测时间即为当前时间
            
            # 1. 始终保存模型性能记录（RMSE + 预测值 + 预测时间）
            # 计算 RMSE（简化版，实际应该基于历史误差计算）
            prediction_error = abs(predicted_cv - current_cv)
            rmse = prediction_error  # 单次预测的 RMSE 即为误差
            
            performance_record = {
                "instance_id": self.instance_id,
                "timestamp": timestamp,
                "model_type": model_type,
                "model_path": self.model_manager.formal_model_path if model_type == "FORMAL" else self.model_manager.shadow_model_path,
                "rmse": rmse,
                "mae": prediction_error,
                "r2": max(0.0, 1.0 - (prediction_error / (self.safety_upper - self.safety_lower + 0.001))),
                "predicted_value": predicted_cv,
                "actual_value": current_cv,
                "prediction_error": prediction_error,
                "predict_time": predict_time,  # ✅ 预测时间
                "sample_count": 1,
                "extra_info": {
                    "control_status": status,
                    "health_score": health_score
                }
            }
            
            self.db.save_model_performance_record(performance_record)
            
            # 2. 如果 MPC 开启，额外保存控制记录（MV + CV target）
            if self.mpc_enabled:
                control_record = {
                    "instance_id": self.instance_id,
                    "controller_name": self.controller_name,
                    "timestamp": timestamp,
                    "mv_value": mv_value,
                    "cv_target": self.target_sp,
                    "cv_actual": current_cv,
                    "predicted_cv": predicted_cv,
                    "control_status": status,
                    "health_score": health_score,
                    "model_used": model_type,
                    "mpc_enabled": True,
                    "extra_info": {
                        "prediction_horizon": self.prediction_horizon,
                        "control_horizon": self.control_horizon,
                        "predict_time": predict_time  # ✅ 预测时间
                    }
                }
                
                self.db.save_mpc_control_record(control_record)
                print(f"[MPC DB] 已保存控制记录 (MPC开启): instance={self.instance_id}, mv={mv_value:.4f}, cv_target={self.target_sp}, predict_time={predict_time}")
            else:
                print(f"[MPC DB] 已保存性能记录 (MPC关闭): instance={self.instance_id}, rmse={rmse:.4f}, pred={predicted_cv:.4f}, predict_time={predict_time}")
            
        except Exception as e:
            print(f"[MPC DB ERROR] 保存数据失败: {e}")
