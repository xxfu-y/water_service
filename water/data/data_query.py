"""
数据查询模块 - 根据标签和时间范围从数据源获取数据
支持多种数据源：数据库、API、文件等
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import numpy as np


class DataQueryService:
    """数据查询服务 - 统一的数据获取接口"""
    
    def __init__(self, config: Dict = None):
        """
        初始化数据查询服务
        
        Args:
            config: 数据源配置字典
        """
        self.config = config or {}
        self.data_sources = {}
        self._register_default_sources()
        print("[DataQuery] 数据查询服务初始化完成")
    
    def _register_default_sources(self):
        """注册默认数据源"""
        # 这里可以注册不同的数据源适配器
        self.data_sources["database"] = self._query_from_database
        self.data_sources["api"] = self._query_from_api
        self.data_sources["file"] = self._query_from_file
        self.data_sources["mock"] = self._query_mock_data
    
    def query_data(self, 
                   data_tags: List[str],
                   target_variable: str,
                   start_time: str,
                   end_time: str,
                   sampling_interval: int = 300,
                   source_type: str = "mock") -> pd.DataFrame:
        """
        根据标签和时间范围查询数据
        
        Args:
            data_tags: 数据标签列表（可解析为多个字段名）
            target_variable: 目标变量字段名
            start_time: 开始时间 (yyyy-MM-dd HH:mm:ss)
            end_time: 结束时间 (yyyy-MM-dd HH:mm:ss)
            sampling_interval: 采样间隔（秒），默认300秒（5分钟）
            source_type: 数据源类型 (database/api/file/mock)
            
        Returns:
            DataFrame: 查询到的数据
        """
        print(f"[DataQuery] 开始查询数据:")
        print(f"  - 数据标签: {data_tags}")
        print(f"  - 目标变量: {target_variable}")
        print(f"  - 时间范围: {start_time} ~ {end_time}")
        print(f"  - 采样间隔: {sampling_interval}秒")
        print(f"  - 数据源: {source_type}")
        
        # 选择数据源
        if source_type not in self.data_sources:
            raise ValueError(f"不支持的数据源类型: {source_type}")
        
        query_func = self.data_sources[source_type]
        
        try:
            # 执行查询
            df = query_func(
                data_tags=data_tags,
                target_variable=target_variable,
                start_time=start_time,
                end_time=end_time,
                sampling_interval=sampling_interval
            )
            
            if df is None or df.empty:
                print(f"[DataQuery WARNING] 未查询到数据")
                return pd.DataFrame()
            
            print(f"[DataQuery] 查询成功，获取 {len(df)} 条记录")
            return df
            
        except Exception as e:
            print(f"[DataQuery ERROR] 查询失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _parse_data_tags(self, data_tags: List[str]) -> List[str]:
        """
        解析数据标签为字段名列表
        
        Args:
            data_tags: 数据标签列表
            
        Returns:
            字段名列表
        """
        field_names = []
        
        for tag in data_tags:
            # 如果标签包含逗号，说明是多个字段的组合
            if "," in tag:
                fields = [f.strip() for f in tag.split(",")]
                field_names.extend(fields)
            else:
                field_names.append(tag)
        
        # 去重
        field_names = list(dict.fromkeys(field_names))
        
        print(f"[DataQuery] 解析标签: {data_tags} -> 字段: {field_names}")
        return field_names
    
    def _query_from_database(self, 
                            data_tags: List[str],
                            target_variable: str,
                            start_time: str,
                            end_time: str,
                            sampling_interval: int) -> pd.DataFrame:
        """
        从 MySQL 数据库查询数据
        
        Args:
            data_tags: 数据标签列表
            target_variable: 目标变量
            start_time: 开始时间
            end_time: 结束时间
            sampling_interval: 采样间隔（秒）
            
        Returns:
            DataFrame: 查询到的数据
        """
        print("[DataQuery] 从 MySQL 数据库查询数据...")
        
        try:
            # 导入 SQLAlchemy
            from sqlalchemy import create_engine, text
            
            # 获取数据库配置
            db_config = self.config.get('database', {})
            
            # 构建数据库连接 URL
            username = db_config.get('username', 'root')
            password = db_config.get('password', '123456')
            host = db_config.get('host', 'localhost')
            port = db_config.get('port', 3306)
            database = db_config.get('database', 'water_db')
            
            # MySQL 连接字符串
            db_url = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}?charset=utf8mb4"
            
            print(f"[DataQuery] 连接数据库: {host}:{port}/{database}")
            
            # 创建数据库引擎
            engine = create_engine(db_url, pool_pre_ping=True)
            
            # 解析标签为字段
            field_names = self._parse_data_tags(data_tags)
            if target_variable and target_variable not in field_names:
                field_names.append(target_variable)
            
            # 构建 SQL 查询
            # 注意：字段名需要用反引号包裹，防止与 MySQL 关键字冲突
            fields_str = ", ".join([f"`{field}`" for field in field_names])
            
            sql = text(f"""
                SELECT {fields_str}, `timestamp`
                FROM `water_sensor_data`
                WHERE `timestamp` >= :start_time 
                  AND `timestamp` <= :end_time
                ORDER BY `timestamp` ASC
            """)
            
            print(f"[DataQuery] 执行 SQL 查询...")
            print(f"  - 字段: {field_names}")
            print(f"  - 时间范围: {start_time} ~ {end_time}")
            
            # 执行查询
            with engine.connect() as conn:
                df = pd.read_sql(
                    sql, 
                    conn,
                    params={
                        'start_time': start_time,
                        'end_time': end_time
                    }
                )
            
            if df.empty:
                print("[DataQuery WARNING] 查询结果为空")
                return pd.DataFrame()
            
            print(f"[DataQuery] 查询成功，获取 {len(df)} 条记录")
            
            # 确保 timestamp 是 datetime 类型
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 如果采样间隔大于0，进行重采样
            if sampling_interval > 0 and len(df) > 0:
                df = df.set_index('timestamp')
                # 使用均值重采样
                freq_str = f"{sampling_interval}s"
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    df_resampled = df[numeric_cols].resample(freq_str).mean()
                    # 保留非数值列（如果有）
                    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
                    if len(non_numeric_cols) > 0:
                        df_non_numeric = df[non_numeric_cols].resample(freq_str).first()
                        df_resampled = pd.concat([df_resampled, df_non_numeric], axis=1)
                    
                    df = df_resampled.reset_index()
                    print(f"[DataQuery] 重采样完成: {len(df)} 条记录 (间隔: {sampling_interval}秒)")
            
            # 关闭引擎
            engine.dispose()
            
            return df
            
        except ImportError:
            print("[DataQuery ERROR] 缺少必要的库: sqlalchemy 或 pymysql")
            print("请安装: pip install sqlalchemy pymysql")
            raise
        except Exception as e:
            print(f"[DataQuery ERROR] 数据库查询失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _query_from_api(self,
                       data_tags: List[str],
                       target_variable: str,
                       start_time: str,
                       end_time: str,
                       sampling_interval: int) -> pd.DataFrame:
        """
        从 API 查询数据
        
        TODO: 实现实际的 API 调用逻辑
        """
        print("[DataQuery] 从 API 查询数据...")
        
        # 示例：使用 requests
        # import requests
        # api_url = self.config.get("api_url")
        # params = {
        #     "tags": ",".join(data_tags),
        #     "target": target_variable,
        #     "start": start_time,
        #     "end": end_time,
        #     "interval": sampling_interval
        # }
        # response = requests.get(api_url, params=params)
        # data = response.json()
        # df = pd.DataFrame(data)
        
        print("[DataQuery WARNING] API 查询功能待实现")
        return pd.DataFrame()
    
    def _query_from_file(self,
                        data_tags: List[str],
                        target_variable: str,
                        start_time: str,
                        end_time: str,
                        sampling_interval: int) -> pd.DataFrame:
        """
        从文件查询数据
        
        TODO: 实现实际的文件读取逻辑
        """
        print("[DataQuery] 从文件查询数据...")
        
        # 示例：从 CSV 或 Parquet 文件读取
        # file_path = self.config.get("data_file_path")
        # df = pd.read_csv(file_path)
        # df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]
        
        print("[DataQuery WARNING] 文件查询功能待实现")
        return pd.DataFrame()
    
    def _query_mock_data(self,
                        data_tags: List[str],
                        target_variable: str,
                        start_time: str,
                        end_time: str,
                        sampling_interval: int) -> pd.DataFrame:
        """
        生成模拟数据（用于测试）
        
        Args:
            data_tags: 数据标签列表
            target_variable: 目标变量
            start_time: 开始时间
            end_time: 结束时间
            sampling_interval: 采样间隔
            
        Returns:
            模拟数据 DataFrame
        """
        print("[DataQuery] 生成模拟数据...")
        
        # 解析时间
        try:
            start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
            end_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            # 尝试 ISO 格式
            start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
        
        # 计算时间点
        time_range = (end_dt - start_dt).total_seconds()
        num_samples = int(time_range / sampling_interval) + 1
        
        if num_samples <= 0:
            print("[DataQuery WARNING] 时间范围无效")
            return pd.DataFrame()
        
        # 解析标签为字段
        field_names = self._parse_data_tags(data_tags)
        if target_variable not in field_names:
            field_names.append(target_variable)
        
        # 生成时间序列
        timestamps = pd.date_range(start=start_dt, end=end_dt, freq=f"{sampling_interval}s")
        
        # 生成模拟数据
        data = {"timestamp": timestamps}
        
        for field in field_names:
            # 根据不同字段生成不同的模拟数据
            if "cod" in field.lower():
                # COD: 50-200
                data[field] = np.random.uniform(50, 200, len(timestamps))
            elif "nh3" in field.lower() or "ammonia" in field.lower():
                # NH3: 5-30
                data[field] = np.random.uniform(5, 30, len(timestamps))
            elif "do" in field.lower():
                # DO: 1-5
                data[field] = np.random.uniform(1, 5, len(timestamps))
            elif "flow" in field.lower() or "rate" in field.lower():
                # Flow: 1000-5000
                data[field] = np.random.uniform(1000, 5000, len(timestamps))
            elif "pressure" in field.lower():
                # Pressure: 0.1-0.5
                data[field] = np.random.uniform(0.1, 0.5, len(timestamps))
            elif "opening" in field.lower() or "valve" in field.lower():
                # Valve opening: 0-100
                data[field] = np.random.uniform(0, 100, len(timestamps))
            elif "mlss" in field.lower():
                # MLSS: 2000-4000
                data[field] = np.random.uniform(2000, 4000, len(timestamps))
            elif "blower" in field.lower():
                if "running" in field.lower():
                    # Blower running: 0 or 1
                    data[field] = np.random.choice([0, 1], len(timestamps))
                elif "opening" in field.lower():
                    data[field] = np.random.uniform(0, 100, len(timestamps))
                elif "pressure" in field.lower():
                    data[field] = np.random.uniform(0.2, 0.6, len(timestamps))
            else:
                # 默认值
                data[field] = np.random.uniform(0, 100, len(timestamps))
        
        df = pd.DataFrame(data)
        
        print(f"[DataQuery] 生成 {len(df)} 条模拟数据")
        return df
    
    def register_custom_source(self, name: str, query_func):
        """
        注册自定义数据源
        
        Args:
            name: 数据源名称
            query_func: 查询函数，签名为:
                query_func(data_tags, target_variable, start_time, end_time, sampling_interval) -> DataFrame
        """
        self.data_sources[name] = query_func
        print(f"[DataQuery] 注册自定义数据源: {name}")


# 导入 numpy（在 mock 数据中使用）
import numpy as np


# 创建全局数据查询服务实例
_query_service = None


def get_query_service(config: Dict = None) -> DataQueryService:
    """
    获取数据查询服务单例
    
    Args:
        config: 数据源配置
        
    Returns:
        DataQueryService 实例
    """
    global _query_service
    if _query_service is None:
        _query_service = DataQueryService(config)
    return _query_service


def reset_query_service():
    """重置数据查询服务（用于测试）"""
    global _query_service
    _query_service = None
