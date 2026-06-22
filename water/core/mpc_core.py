"""
MPC (Model Predictive Control) 控制器核心逻辑
实现基于预测模型的最优控制算法
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple


class MpcController:
    """MPC 控制器类"""
    
    def __init__(self, config: Dict):
        """
        初始化 MPC 控制器
        
        Args:
            config: MPC 控制器配置字典
        """
        self.config = config
        self.controller_name = config.get("controllerName", "unknown")
        self.instance_name = config.get("instanceName", "unknown")
        
        # 基本参数
        self.sampling_time = config.get("samplingTime", 300.0)  # 采样时间（秒）
        self.prediction_horizon = config.get("predictionHorizon", 10)  # 预测时域
        self.control_horizon = config.get("controlHorizon", 5)  # 控制时域
        
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
        
        # 历史数据
        self.history_data = []
        self.control_history = []
        
        print(f"[MPC] 控制器 {self.controller_name} 初始化完成")
        print(f"  - 实例: {self.instance_name}")
        print(f"  - 采样时间: {self.sampling_time}s")
        print(f"  - 预测时域: {self.prediction_horizon}")
        print(f"  - 控制时域: {self.control_horizon}")
        print(f"  - 目标值: {self.target_sp}")
    
    def compute_control(self, sensor_data: List[Dict], model_predict_func=None) -> Dict:
        """
        计算最优控制量
        
        Args:
            sensor_data: 传感器数据列表
            model_predict_func: 预测模型函数，输入历史数据，返回预测值
            
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
                    message="CV 在死区内，保持当前 MV"
                )
            
            # 3. 使用预测模型进行多步预测
            if model_predict_func is None:
                # 如果没有提供预测模型，使用简化策略
                mv_value = self._simple_control(current_cv)
                predicted_cv = current_cv
            else:
                # 使用 MPC 优化算法
                mv_value, predicted_cv = self._mpc_optimization(
                    sensor_data, 
                    model_predict_func
                )
            
            # 4. 应用 MV 约束
            mv_value = self._apply_mv_constraints(mv_value)
            
            # 5. 检查安全性
            if not self._check_safety(predicted_cv):
                return self._create_response(
                    mv_value=mv_value,
                    predicted_cv=predicted_cv,
                    status="ERROR",
                    message="预测值超出安全范围"
                )
            
            # 6. 计算健康度
            health_score = self._calculate_health_score(current_cv, predicted_cv)
            
            # 7. 记录历史
            self._record_control(current_cv, mv_value, predicted_cv)
            
            # 8. 确定控制状态
            status = self._determine_status(current_cv, mv_value)
            
            return self._create_response(
                mv_value=mv_value,
                predicted_cv=predicted_cv,
                status=status,
                health_score=health_score,
                message="控制计算成功"
            )
            
        except Exception as e:
            print(f"[MPC ERROR] 控制计算失败: {e}")
            import traceback
            traceback.print_exc()
            
            return self._create_response(
                mv_value=0.0,
                predicted_cv=0.0,
                status="ERROR",
                message=f"控制计算失败: {str(e)}"
            )
    
    def _get_current_cv(self, sensor_data: List[Dict]) -> float:
        """获取当前 CV 值"""
        if not sensor_data:
            return self.target_sp
        
        # 从最新的传感器数据中获取 CV 值
        latest_data = sensor_data[-1]
        values = latest_data.get("values", {})
        
        # 查找 CV 特征
        for binding in self.feature_bindings:
            if binding.get("type") == "CV":
                feature_name = binding.get("featureName")
                if feature_name in values:
                    return float(values[feature_name])
        
        # 如果没找到，返回默认值
        return self.target_sp
    
    def _get_last_mv(self) -> float:
        """获取上一个 MV 值"""
        if self.control_history:
            return self.control_history[-1]["mv_value"]
        return self.target_sp  # 默认返回目标值
    
    def _simple_control(self, current_cv: float) -> float:
        """简化控制策略（PID 风格）"""
        error = self.target_sp - current_cv
        
        # 简化的 P 控制
        kp = 2.0  # 比例增益
        mv_change = kp * error
        
        # 限制变化率
        mv_change = np.clip(mv_change, -self.mv_max_change, self.mv_max_change)
        
        last_mv = self._get_last_mv()
        new_mv = last_mv + mv_change
        
        return new_mv
    
    def _mpc_optimization(self, sensor_data: List[Dict], 
                         model_predict_func) -> Tuple[float, float]:
        """
        MPC 优化算法
        
        Args:
            sensor_data: 传感器数据
            model_predict_func: 预测模型函数
            
        Returns:
            Tuple[float, float]: (最优 MV, 预测 CV)
        """
        # 这里实现完整的 MPC 优化算法
        # 由于需要求解优化问题，这里提供一个简化版本
        
        best_mv = self._get_last_mv()
        best_cost = float('inf')
        best_predicted_cv = self.target_sp
        
        # 在 MV 范围内搜索最优解
        mv_candidates = np.linspace(
            self.mv_physical_lower,
            self.mv_physical_upper,
            20  # 候选点数量
        )
        
        for mv_candidate in mv_candidates:
            # 使用预测模型预测未来轨迹
            try:
                predicted_trajectory = model_predict_func(
                    sensor_data, 
                    mv_candidate,
                    self.prediction_horizon
                )
                
                # 计算代价函数
                cost = self._calculate_cost(predicted_trajectory, mv_candidate)
                
                if cost < best_cost:
                    best_cost = cost
                    best_mv = mv_candidate
                    best_predicted_cv = predicted_trajectory[-1] if predicted_trajectory else self.target_sp
                    
            except Exception as e:
                print(f"[MPC] 预测失败: {e}")
                continue
        
        return best_mv, best_predicted_cv
    
    def _calculate_cost(self, predicted_trajectory: List[float], 
                       mv_value: float) -> float:
        """
        计算代价函数
        
        Args:
            predicted_trajectory: 预测轨迹
            mv_value: MV 值
            
        Returns:
            float: 代价值
        """
        if not predicted_trajectory:
            return float('inf')
        
        # 跟踪误差
        tracking_error = sum((y - self.target_sp) ** 2 for y in predicted_trajectory)
        
        # MV 变化惩罚
        last_mv = self._get_last_mv()
        mv_change_penalty = 0.1 * (mv_value - last_mv) ** 2
        
        # 总代价
        total_cost = tracking_error + mv_change_penalty
        
        return total_cost
    
    def _apply_mv_constraints(self, mv_value: float) -> float:
        """应用 MV 约束"""
        # 物理限制
        mv_value = np.clip(mv_value, self.mv_physical_lower, self.mv_physical_upper)
        
        # 变化率限制
        last_mv = self._get_last_mv()
        max_change = self.mv_max_change
        min_change = -self.mv_max_change
        
        mv_change = mv_value - last_mv
        mv_change = np.clip(mv_change, min_change, max_change)
        
        return last_mv + mv_change
    
    def _check_safety(self, predicted_cv: float) -> bool:
        """检查安全性"""
        return self.safety_lower <= predicted_cv <= self.safety_upper
    
    def _calculate_health_score(self, current_cv: float, 
                               predicted_cv: float) -> float:
        """
        计算健康度评分
        
        Returns:
            float: 健康度评分 (0-1)
        """
        # 基于预测误差计算健康度
        error = abs(predicted_cv - self.target_sp)
        max_error = self.safety_upper - self.safety_lower
        
        if max_error == 0:
            return 1.0
        
        # 健康度 = 1 - (误差 / 最大允许误差)
        health_score = max(0.0, 1.0 - (error / max_error))
        
        return health_score
    
    def _record_control(self, cv_value: float, mv_value: float, 
                       predicted_cv: float):
        """记录控制历史"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "cv_value": cv_value,
            "mv_value": mv_value,
            "predicted_cv": predicted_cv
        }
        self.control_history.append(record)
        
        # 保留最近 1000 条记录
        if len(self.control_history) > 1000:
            self.control_history = self.control_history[-1000:]
    
    def _determine_status(self, current_cv: float, mv_value: float) -> str:
        """确定控制状态"""
        # 检查是否饱和
        if mv_value >= self.mv_physical_upper * 0.95 or \
           mv_value <= self.mv_physical_lower * 1.05:
            return "SATURATED"
        
        # 检查是否在死区
        if self.deadband_lower <= current_cv <= self.deadband_upper:
            return "DEADBAND"
        
        return "NORMAL"
    
    def _create_response(self, mv_value: float, predicted_cv: float,
                        status: str, message: str, 
                        health_score: float = 1.0) -> Dict:
        """创建响应字典"""
        return {
            "success": status != "ERROR",
            "message": message,
            "controller_name": self.controller_name,
            "mv_value": float(mv_value),
            "predicted_cv": float(predicted_cv),
            "control_status": status,
            "health_score": float(health_score),
            "timestamp": datetime.now().isoformat()
        }
    
    def update_config(self, new_config: Dict):
        """更新控制器配置"""
        self.config.update(new_config)
        print(f"[MPC] 控制器 {self.controller_name} 配置已更新")
    
    def get_status(self) -> Dict:
        """获取控制器状态"""
        return {
            "controller_name": self.controller_name,
            "instance_name": self.instance_name,
            "status": "RUNNING",
            "last_mv": self._get_last_mv(),
            "history_count": len(self.control_history),
            "timestamp": datetime.now().isoformat()
        }
