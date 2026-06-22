"""
数据适配器模块 - 统一 Predict 和 MPC 服务的数据获取与转换逻辑
"""

import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime


class DataAdapter:
    """数据适配器 - 在不同数据格式之间转换"""
    
    @staticmethod
    def water_sensor_data_to_dataframe(sensor_data_list: List[Dict]) -> pd.DataFrame:
        """
        将 WaterSensorData 列表转换为 DataFrame（用于 Predict 服务）
        
        Args:
            sensor_data_list: WaterSensorData 字典列表
            
        Returns:
            DataFrame
        """
        if not sensor_data_list:
            return pd.DataFrame()
        
        rows = []
        for data in sensor_data_list:
            rows.append({
                "id": data.get("id"),
                "timestamp": data.get("timestamp"),
                "inflow_rate2": data.get("inflow_rate2", 0.0),
                "inflow_cod": data.get("inflow_cod", 0.0),
                "inflow_nh3": data.get("inflow_nh3", 0.0),
                "outflow_cod": data.get("outflow_cod", 0.0),
                "outflow_nh3": data.get("outflow_nh3", 0.0),
                "bio2_front_aeration_flow_meas": data.get("bio2_front_aeration_flow_meas", 0.0),
                "bio2_front_valve_opening_meas": data.get("bio2_front_valve_opening_meas", 0.0),
                "bio2_end_aeration_flow_meas": data.get("bio2_end_aeration_flow_meas", 0.0),
                "bio2_end_valve_opening_meas": data.get("bio2_end_valve_opening_meas", 0.0),
                "bio2_front_do_meas": data.get("bio2_front_do_meas", 0.0),
                "bio2_end_do_meas": data.get("bio2_end_do_meas", 0.0),
                "bio2_mlss_meas": data.get("bio2_mlss_meas", 0.0),
                "bio2_aeration_pipe_pressure_meas": data.get("bio2_aeration_pipe_pressure_meas", 0.0),
                "bio2_1_blower_opening_meas": data.get("bio2_1_blower_opening_meas", 0.0),
                "bio2_1_blower_is_running": data.get("bio2_1_blower_is_running", 0),
                "bio2_1_blower_pressure_meas": data.get("bio2_1_blower_pressure_meas", 0.0),
                "bio2_2_blower_opening_meas": data.get("bio2_2_blower_opening_meas", 0.0),
                "bio2_2_blower_is_running": data.get("bio2_2_blower_is_running", 0),
                "bio2_2_blower_pressure_meas": data.get("bio2_2_blower_pressure_meas", 0.0)
            })
        
        return pd.DataFrame(rows)
    
    @staticmethod
    def sensor_data_to_dict_list(sensor_data_list: List[Dict]) -> List[Dict]:
        """
        将 SensorData 列表转换为字典列表（用于 MPC 服务）
        
        Args:
            sensor_data_list: SensorData 字典列表，每个包含 timestamp 和 values
            
        Returns:
            字典列表
        """
        if not sensor_data_list:
            return []
        
        result = []
        for data in sensor_data_list:
            result.append({
                "timestamp": data.get("timestamp", datetime.now().isoformat()),
                "values": data.get("values", {})
            })
        
        return result
    
    @staticmethod
    def dataframe_to_sensor_data(df: pd.DataFrame) -> List[Dict]:
        """
        将 DataFrame 转换为 SensorData 格式（用于 MPC 服务）
        
        Args:
            df: DataFrame
            
        Returns:
            SensorData 字典列表
        """
        if df is None or df.empty:
            return []
        
        result = []
        for _, row in df.iterrows():
            # 提取所有数值列作为 values
            values = {}
            for col in df.columns:
                if col != "id" and col != "timestamp":
                    try:
                        values[col] = float(row[col])
                    except (ValueError, TypeError):
                        continue
            
            result.append({
                "timestamp": str(row.get("timestamp", datetime.now().isoformat())),
                "values": values
            })
        
        return result
    
    @staticmethod
    def extract_features_for_mpc(sensor_data: List[Dict], 
                                 feature_bindings: List[Dict]) -> Dict[str, float]:
        """
        根据特征绑定提取 MPC 需要的特征
        
        Args:
            sensor_data: 传感器数据列表
            feature_bindings: 特征绑定配置
            
        Returns:
            特征字典 {feature_name: value}
        """
        if not sensor_data or not feature_bindings:
            return {}
        
        # 使用最新的数据点
        latest_data = sensor_data[-1]
        values = latest_data.get("values", {})
        
        features = {}
        for binding in feature_bindings:
            feature_name = binding.get("featureName")
            tag_name = binding.get("tagName")
            
            # 优先使用 tagName，其次使用 featureName
            key = tag_name if tag_name else feature_name
            
            if key in values:
                features[feature_name] = float(values[key])
        
        return features
    
    @staticmethod
    def validate_sensor_data(data: List[Dict], required_fields: List[str] = None) -> bool:
        """
        验证传感器数据的有效性
        
        Args:
            data: 传感器数据列表
            required_fields: 必需的字段列表
            
        Returns:
            是否有效
        """
        if not data:
            return False
        
        # 检查是否有数据
        if len(data) == 0:
            return False
        
        # 检查必需字段
        if required_fields:
            latest_data = data[-1]
            if isinstance(latest_data, dict):
                if "values" in latest_data:
                    values = latest_data["values"]
                    for field in required_fields:
                        if field not in values:
                            return False
        
        return True
    
    @staticmethod
    def get_latest_value(sensor_data: List[Dict], feature_name: str, 
                        default: float = 0.0) -> float:
        """
        获取最新数据点的某个特征值
        
        Args:
            sensor_data: 传感器数据列表
            feature_name: 特征名称
            default: 默认值
            
        Returns:
            特征值
        """
        if not sensor_data:
            return default
        
        latest_data = sensor_data[-1]
        values = latest_data.get("values", {})
        
        return float(values.get(feature_name, default))


class DataSourceManager:
    """数据源管理器 - 统一管理数据获取"""
    
    def __init__(self):
        self.adapters = {}
    
    def register_adapter(self, name: str, adapter_func):
        """
        注册数据适配器
        
        Args:
            name: 适配器名称
            adapter_func: 适配函数
        """
        self.adapters[name] = adapter_func
        print(f"[DataSource] 注册数据适配器: {name}")
    
    def fetch_and_convert(self, source_name: str, target_format: str = "dataframe"):
        """
        从指定数据源获取数据并转换
        
        Args:
            source_name: 数据源名称
            target_format: 目标格式 (dataframe/sensor_data/dict_list)
            
        Returns:
            转换后的数据
        """
        if source_name not in self.adapters:
            raise ValueError(f"未找到数据源: {source_name}")
        
        # 获取原始数据
        raw_data = self.adapters[source_name]()
        
        if raw_data is None:
            return None
        
        # 转换为目标格式
        adapter = DataAdapter()
        
        if target_format == "dataframe":
            if isinstance(raw_data, list):
                return adapter.water_sensor_data_to_dataframe(raw_data)
            elif isinstance(raw_data, pd.DataFrame):
                return raw_data
        
        elif target_format == "sensor_data":
            if isinstance(raw_data, pd.DataFrame):
                return adapter.dataframe_to_sensor_data(raw_data)
            elif isinstance(raw_data, list):
                return adapter.sensor_data_to_dict_list(raw_data)
        
        elif target_format == "dict_list":
            if isinstance(raw_data, list):
                return raw_data
        
        return raw_data


# 示例：创建通用的数据获取函数
def create_data_fetcher_from_database(db_config: Dict):
    """
    从数据库创建数据获取器
    
    Args:
        db_config: 数据库配置
        
    Returns:
        数据获取函数
    """
    def fetch_from_db():
        """从数据库获取数据"""
        # TODO: 实现实际的数据库查询逻辑
        # 这里需要根据实际的数据库类型和表结构调整
        pass
    
    return fetch_from_db


def create_data_fetcher_from_api(api_url: str, api_key: str = None):
    """
    从 API 创建数据获取器
    
    Args:
        api_url: API URL
        api_key: API 密钥
        
    Returns:
        数据获取函数
    """
    def fetch_from_api():
        """从 API 获取数据"""
        # TODO: 实现实际的 API 调用逻辑
        import requests
        
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        try:
            response = requests.get(api_url, headers=headers, timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"[API Fetch ERROR] 获取数据失败: {e}")
            return None
    
    return fetch_from_api


def create_data_fetcher_from_message_queue(mq_config: Dict):
    """
    从消息队列创建数据获取器
    
    Args:
        mq_config: 消息队列配置
        
    Returns:
        数据获取函数
    """
    def fetch_from_mq():
        """从消息队列获取数据"""
        # TODO: 实现实际的消息队列消费逻辑
        pass
    
    return fetch_from_mq
