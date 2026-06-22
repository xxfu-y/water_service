"""
数据仓库历史数据 gRPC 客户端
用于从 DwHistoryStreamService 获取历史数据
"""

import grpc
import pandas as pd
from typing import List, Optional, Dict
from collections import defaultdict
from water.utils.config import get_external_service_address


class DwHistoryClient:
    """数据仓库历史数据客户端"""
    
    def __init__(self, server_address: str = None):
        """
        初始化数据仓库客户端
        
        Args:
            server_address: 数据仓库 gRPC 服务地址，如果为 None 则使用配置文件的默认地址
        """
        if server_address is None:
            # 从配置文件获取默认地址
            self.server_address = get_external_service_address("dw_history_service")
        else:
            self.server_address = server_address
        self.channel = None
        self.stub = None
    
    def _connect(self):
        """建立 gRPC 连接"""
        if self.channel is None:
            self.channel = grpc.insecure_channel(self.server_address)
            # 注意：需要导入数据仓库的 proto 生成的代码
            # 这里假设已经生成了 dw_history_pb2 和 dw_history_pb2_grpc
            try:
                import water.proto.dw_history_pb2 as dw_pb2
                import water.proto.dw_history_pb2_grpc as dw_grpc
                self.stub = dw_grpc.DwHistoryStreamServiceStub(self.channel)
                self.pb2 = dw_pb2
            except ImportError as e:
                print(f"[DwHistoryClient ERROR] 无法导入数据仓库 proto 文件: {e}")
                print("请先编译 proto 文件: python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. dw_history.proto")
                raise
    
    def _disconnect(self):
        """断开 gRPC 连接"""
        if self.channel is not None:
            self.channel.close()
            self.channel = None
            self.stub = None
    
    def query_history(self, 
                     tags: List[str],
                     start_time: str,
                     end_time: str,
                     interval: str = "5m",
                     batch_size: int = 5000) -> Optional[pd.DataFrame]:
        """
        查询历史数据（流式）
        
        Args:
            tags: 标签列表，如 ["inflow_cod", "inflow_rate1"]
            start_time: 开始时间，格式 "yyyy-MM-dd HH:mm:ss"
            end_time: 结束时间，格式 "yyyy-MM-dd HH:mm:ss"
            interval: 聚合间隔，可选值: 1s/1m/3m/5m/15m/30m/1h
            batch_size: 每批条数，默认 5000，最大 20000
            
        Returns:
            DataFrame: 查询到的数据，失败返回 None
        """
        print(f"[DwHistoryClient] 开始查询历史数据:")
        print(f"  - 标签: {tags}")
        print(f"  - 时间范围: {start_time} ~ {end_time}")
        print(f"  - 聚合间隔: {interval}")
        print(f"  - 批次大小: {batch_size}")
        
        try:
            # 1. 建立连接
            self._connect()
            
            # 2. 构建请求
            request = self.pb2.HistoryRequest(
                tags=tags,
                start_time=start_time,
                end_time=end_time,
                interval=interval,
                batch_size=batch_size
            )
            
            # 3. 流式接收数据
            print("[DwHistoryClient] 开始接收数据流...")
            
            # 使用字典存储每个标签的数据点 {tag: [(time, value), ...]}
            tag_data = defaultdict(list)
            total_points = 0
            batch_count = 0
            
            for batch in self.stub.QueryHistory(request):
                batch_count += 1
                points_in_batch = len(batch.points)
                total_points += points_in_batch
                
                print(f"[DwHistoryClient] 收到第 {batch.batch_seq} 批，共 {points_in_batch} 条数据")
                
                # 处理每一批数据点
                for pt in batch.points:
                    # 只处理 ONLINE 状态的数据
                    if pt.status == "ONLINE":
                        tag_data[pt.tag].append({
                            'time': pt.time,
                            'value': pt.value
                        })
                
                # 如果还有更多数据，继续接收
                if not batch.has_more:
                    print("[DwHistoryClient] 数据传输完成")
                    break
            
            print(f"[DwHistoryClient] 总共接收 {total_points} 条数据点，{batch_count} 个批次")
            
            # 4. 将数据转换为 DataFrame（宽表格式）
            if not tag_data:
                print("[DwHistoryClient WARNING] 未接收到任何数据")
                return pd.DataFrame()
            
            print(f"[DwHistoryClient] 正在转换数据为 DataFrame...")
            print(f"  - 标签数量: {len(tag_data)}")
            
            # 将所有时间点和对应的值组织成 DataFrame
            # 首先收集所有唯一的时间点
            all_times = set()
            for tag, points in tag_data.items():
                for point in points:
                    all_times.add(point['time'])
            
            all_times = sorted(all_times)
            print(f"  - 时间点数量: {len(all_times)}")
            
            # 创建时间索引
            df = pd.DataFrame({'timestamp': all_times})
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            
            # 为每个标签创建列
            for tag, points in tag_data.items():
                # 创建该标签的时间-值映射
                time_value_map = {pt['time']: pt['value'] for pt in points}
                
                # 填充到 DataFrame
                values = [time_value_map.get(time, None) for time in all_times]
                df[tag] = values
            
            # 重置索引，使 timestamp 成为普通列
            df = df.reset_index()
            
            # 前向填充缺失值（如果需要）
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                df[numeric_cols] = df[numeric_cols].ffill().bfill()
            
            print(f"[DwHistoryClient] 数据转换完成:")
            print(f"  - DataFrame shape: {df.shape}")
            print(f"  - 列名: {list(df.columns)}")
            
            return df
            
        except grpc.RpcError as e:
            print(f"[DwHistoryClient ERROR] gRPC 调用失败: {e.code()}: {e.details()}")
            return None
        except Exception as e:
            print(f"[DwHistoryClient ERROR] 查询失败: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            # 5. 断开连接
            self._disconnect()
    
    def close(self):
        """关闭客户端连接"""
        self._disconnect()


# 便捷函数
def query_dw_history(tags: List[str],
                    start_time: str,
                    end_time: str,
                    interval: str = "5m",
                    batch_size: int = 5000,
                    server_address: str = None) -> Optional[pd.DataFrame]:
    """
    便捷函数：查询数据仓库历史数据
    
    Args:
        tags: 标签列表
        start_time: 开始时间
        end_time: 结束时间
        interval: 聚合间隔
        batch_size: 批次大小
        server_address: 服务地址，如果为 None 则使用配置文件的默认地址
        
    Returns:
        DataFrame: 查询到的数据
    """
    client = DwHistoryClient(server_address)
    try:
        return client.query_history(tags, start_time, end_time, interval, batch_size)
    finally:
        client.close()
