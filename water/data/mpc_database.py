"""
MPC 数据持久化模块
负责将控制数据、预测数据、模型性能指标存储到数据库
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path


class MpcDatabase:
    """MPC 数据库管理器"""
    
    def __init__(self, db_path: str = "mpc_data.db"):
        """
        初始化数据库
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self.conn = None
        self._init_database()
        print(f"[MPC DB] 数据库初始化完成: {db_path}")
    
    def _init_database(self):
        """初始化数据库表结构"""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.execute("PRAGMA journal_mode=WAL")  # 提高并发性能
            
            cursor = self.conn.cursor()
            
            # 1. MPC 控制记录表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS mpc_control_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    instance_id TEXT NOT NULL,
                    controller_name TEXT,
                    timestamp TEXT NOT NULL,
                    
                    -- MPC 开启时的数据
                    mv_value REAL,                    -- 计算值 MV
                    cv_target REAL,                   -- 目标值 CV (target_sp)
                    cv_actual REAL,                   -- 实际 CV 值
                    predicted_cv REAL,                -- 预测 CV 值
                    
                    -- 控制状态
                    control_status TEXT,              -- NORMAL/DEADBAND/SATURATED/ERROR
                    health_score REAL,                -- 健康度评分
                    model_used TEXT,                  -- FORMAL/SHADOW/SIMPLIFIED
                    
                    -- 元数据
                    mpc_enabled INTEGER DEFAULT 0,    -- 是否开启 MPC
                    extra_info TEXT                   -- JSON 格式的额外信息
                )
            """)
            
            # 2. 模型性能记录表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_performance_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    instance_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    
                    -- 模型标识
                    model_type TEXT NOT NULL,         -- FORMAL/SHADOW
                    model_path TEXT,                  -- 模型路径
                    
                    -- 性能指标
                    rmse REAL,                        -- RMSE
                    mae REAL,                         -- MAE
                    r2 REAL,                          -- R²
                    
                    -- 预测数据
                    predicted_value REAL,             -- 预测值
                    actual_value REAL,                -- 实际值
                    prediction_error REAL,            -- 预测误差
                    predict_time TEXT,                -- 预测时间（ISO 8601格式）
                    
                    -- 元数据
                    sample_count INTEGER DEFAULT 0,   -- 样本数量
                    extra_info TEXT                   -- JSON 格式的额外信息
                )
            """)
            
            # 3. 实例配置表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS instance_configs (
                    instance_id TEXT PRIMARY KEY,
                    controller_name TEXT,
                    config_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    is_active INTEGER DEFAULT 1
                )
            """)
            
            # 创建索引以提高查询性能
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_mpc_instance_time 
                ON mpc_control_records(instance_id, timestamp)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_instance_time 
                ON model_performance_records(instance_id, timestamp)
            """)
            
            self.conn.commit()
            print("[MPC DB] 数据库表结构创建成功")
            
        except Exception as e:
            print(f"[MPC DB ERROR] 数据库初始化失败: {e}")
            raise
    
    def save_mpc_control_record(self, record: Dict):
        """
        保存 MPC 控制记录
        
        Args:
            record: 控制记录字典，包含以下字段：
                - instance_id: 实例 ID
                - controller_name: 控制器名称
                - timestamp: 时间戳
                - mv_value: 计算值 MV
                - cv_target: 目标值 CV
                - cv_actual: 实际 CV 值
                - predicted_cv: 预测 CV 值
                - control_status: 控制状态
                - health_score: 健康度评分
                - model_used: 使用的模型
                - mpc_enabled: 是否开启 MPC
                - extra_info: 额外信息（字典）
        """
        try:
            cursor = self.conn.cursor()
            
            # 序列化 extra_info
            extra_info_json = json.dumps(record.get("extra_info", {})) if record.get("extra_info") else None
            
            cursor.execute("""
                INSERT INTO mpc_control_records (
                    instance_id, controller_name, timestamp,
                    mv_value, cv_target, cv_actual, predicted_cv,
                    control_status, health_score, model_used,
                    mpc_enabled, extra_info
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.get("instance_id"),
                record.get("controller_name"),
                record.get("timestamp", datetime.now().isoformat()),
                record.get("mv_value"),
                record.get("cv_target"),
                record.get("cv_actual"),
                record.get("predicted_cv"),
                record.get("control_status"),
                record.get("health_score"),
                record.get("model_used"),
                1 if record.get("mpc_enabled", False) else 0,
                extra_info_json
            ))
            
            self.conn.commit()
            
        except Exception as e:
            print(f"[MPC DB ERROR] 保存控制记录失败: {e}")
            self.conn.rollback()
    
    def save_model_performance_record(self, record: Dict):
        """
        保存模型性能记录
        
        Args:
            record: 性能记录字典，包含以下字段：
                - instance_id: 实例 ID
                - timestamp: 时间戳
                - model_type: 模型类型 (FORMAL/SHADOW)
                - model_path: 模型路径
                - rmse: RMSE
                - mae: MAE
                - r2: R²
                - predicted_value: 预测值
                - actual_value: 实际值
                - prediction_error: 预测误差
                - sample_count: 样本数量
                - extra_info: 额外信息（字典）
        """
        try:
            cursor = self.conn.cursor()
            
            # 序列化 extra_info
            extra_info_json = json.dumps(record.get("extra_info", {})) if record.get("extra_info") else None
            
            cursor.execute("""
                INSERT INTO model_performance_records (
                    instance_id, timestamp, model_type, model_path,
                    rmse, mae, r2,
                    predicted_value, actual_value, prediction_error, predict_time,
                    sample_count, extra_info
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.get("instance_id"),
                record.get("timestamp", datetime.now().isoformat()),
                record.get("model_type"),
                record.get("model_path"),
                record.get("rmse"),
                record.get("mae"),
                record.get("r2"),
                record.get("predicted_value"),
                record.get("actual_value"),
                record.get("prediction_error"),
                record.get("predict_time"),
                record.get("sample_count", 1),
                extra_info_json
            ))
            
            self.conn.commit()
            
        except Exception as e:
            print(f"[MPC DB ERROR] 保存性能记录失败: {e}")
            self.conn.rollback()
    
    def save_instance_config(self, instance_id: str, config: Dict):
        """
        保存实例配置
        
        Args:
            instance_id: 实例 ID
            config: 配置字典
        """
        try:
            cursor = self.conn.cursor()
            now = datetime.now().isoformat()
            config_json = json.dumps(config)
            
            cursor.execute("""
                INSERT OR REPLACE INTO instance_configs (
                    instance_id, controller_name, config_json,
                    created_at, updated_at, is_active
                ) VALUES (?, ?, ?, ?, ?, 1)
            """, (
                instance_id,
                config.get("controllerName"),
                config_json,
                now,
                now
            ))
            
            self.conn.commit()
            
        except Exception as e:
            print(f"[MPC DB ERROR] 保存实例配置失败: {e}")
            self.conn.rollback()
    
    def get_recent_control_records(self, instance_id: str, limit: int = 100) -> List[Dict]:
        """
        获取最近的控制记录
        
        Args:
            instance_id: 实例 ID
            limit: 返回记录数量
            
        Returns:
            List[Dict]: 控制记录列表
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT * FROM mpc_control_records
                WHERE instance_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (instance_id, limit))
            
            columns = [desc[0] for desc in cursor.description]
            records = []
            
            for row in cursor.fetchall():
                record = dict(zip(columns, row))
                # 反序列化 extra_info
                if record.get("extra_info"):
                    record["extra_info"] = json.loads(record["extra_info"])
                records.append(record)
            
            return records
            
        except Exception as e:
            print(f"[MPC DB ERROR] 查询控制记录失败: {e}")
            return []
    
    def get_model_performance_stats(self, instance_id: str, model_type: str, 
                                   days: int = 7) -> Dict:
        """
        获取模型性能统计
        
        Args:
            instance_id: 实例 ID
            model_type: 模型类型 (FORMAL/SHADOW)
            days: 统计天数
            
        Returns:
            Dict: 性能统计结果
        """
        try:
            cursor = self.conn.cursor()
            
            # 计算起始时间
            from datetime import timedelta
            start_time = (datetime.now() - timedelta(days=days)).isoformat()
            
            cursor.execute("""
                SELECT 
                    AVG(rmse) as avg_rmse,
                    AVG(mae) as avg_mae,
                    AVG(r2) as avg_r2,
                    COUNT(*) as sample_count,
                    MIN(timestamp) as first_record,
                    MAX(timestamp) as last_record
                FROM model_performance_records
                WHERE instance_id = ? 
                  AND model_type = ?
                  AND timestamp >= ?
            """, (instance_id, model_type, start_time))
            
            row = cursor.fetchone()
            if row:
                return {
                    "avg_rmse": row[0],
                    "avg_mae": row[1],
                    "avg_r2": row[2],
                    "sample_count": row[3],
                    "first_record": row[4],
                    "last_record": row[5]
                }
            
            return {}
            
        except Exception as e:
            print(f"[MPC DB ERROR] 查询性能统计失败: {e}")
            return {}
    
    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            print("[MPC DB] 数据库连接已关闭")
    
    def __del__(self):
        """析构函数"""
        self.close()
