"""
CSV 文件服务客户端 - 用于从 WaterCsvFileService 下载和解析 CSV 文件
"""

import grpc
import pandas as pd
import os
from typing import Optional
import tempfile
import shutil

# 导入 gRPC 生成的代码
import water.proto.water_pb2 as water_pb2
import water.proto.water_pb2_grpc as water_pb2_grpc
from water.utils.config import get_port, GRPC_HOST


class CsvFileClient:
    """CSV 文件客户端 - 提供 CSV 文件的下载和解析功能"""
    
    def __init__(self, server_address: str = None):
        """
        初始化 CSV 文件客户端
        
        Args:
            server_address: WaterCsvFileService 的服务地址，如果为 None 则使用配置文件的默认地址
        """
        if server_address is None:
            # 从配置文件获取默认地址
            port = get_port("csv_file_service")
            self.server_address = f"{GRPC_HOST}:{port}"
        else:
            self.server_address = server_address
        self.channel = None
        self.stub = None
    
    def _connect(self):
        """建立 gRPC 连接"""
        if self.channel is None:
            self.channel = grpc.insecure_channel(self.server_address)
            self.stub = water_pb2_grpc.WaterCsvFileServiceStub(self.channel)
    
    def _disconnect(self):
        """断开 gRPC 连接"""
        if self.channel is not None:
            self.channel.close()
            self.channel = None
            self.stub = None
    
    def download_and_parse_csv(self, dataset_name: str, 
                               temp_dir: Optional[str] = None,
                               keep_file: bool = False) -> Optional[pd.DataFrame]:
        """
        下载 CSV 文件并解析为 DataFrame
        
        Args:
            dataset_name: 数据集名称（CSV 文件名）
            temp_dir: 临时目录路径，如果为 None 则使用系统临时目录
            keep_file: 是否在解析后保留文件（默认 False，即删除文件）
            
        Returns:
            DataFrame: 解析后的数据，失败返回 None
        """
        print(f"[CsvFileClient] 开始下载数据集: {dataset_name}")
        
        # 确保临时目录存在
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp(prefix="water_csv_")
            cleanup_temp = True  # 标记需要清理
        else:
            os.makedirs(temp_dir, exist_ok=True)
            cleanup_temp = not keep_file
        
        csv_path = os.path.join(temp_dir, dataset_name)
        
        try:
            # 1. 建立连接
            self._connect()
            
            # 2. 调用 gRPC 服务上传/获取 CSV 文件
            # 注意：根据 proto 定义，UploadCsvFile 接收 csv_name 并返回文件信息
            # 这里假设服务端会将文件内容返回或在本地可访问
            request = water_pb2.CsvFileUploadRequest(csv_name=dataset_name)
            response = self.stub.UploadCsvFile(request)
            
            if not response.success:
                print(f"[CsvFileClient ERROR] 下载失败: {response.message}")
                return None
            
            print(f"[CsvFileClient] 文件信息:")
            print(f"  - 文件大小: {response.file_size} bytes")
            print(f"  - 数据行数: {response.row_count}")
            
            # 3. 检查文件是否存在于本地（假设服务端已将文件保存到指定位置）
            # 如果文件不在本地，需要从响应中获取文件内容
            # 这里需要根据实际的 WaterCsvFileService 实现来调整
            
            # 方案 A: 如果文件已经保存在某个标准位置
            standard_paths = [
                f"data/datasets/{dataset_name}",
                f"datasets/{dataset_name}",
                dataset_name
            ]
            
            file_found = False
            for path in standard_paths:
                if os.path.exists(path):
                    csv_path = path
                    file_found = True
                    print(f"[CsvFileClient] 找到文件: {path}")
                    break
            
            # 方案 B: 如果需要通过 gRPC 传输文件内容
            # 这需要修改 proto 添加文件内容字段到响应中
            if not file_found:
                print(f"[CsvFileClient WARNING] 文件未在标准路径找到，尝试从响应获取")
                # TODO: 如果 response 包含 file_content，则写入临时文件
                # if hasattr(response, 'file_content') and response.file_content:
                #     with open(csv_path, 'wb') as f:
                #         f.write(response.file_content)
                #     file_found = True
            
            if not file_found:
                print(f"[CsvFileClient ERROR] 无法找到或获取文件: {dataset_name}")
                return None
            
            # 4. 解析 CSV 文件（使用分号分隔符）
            print(f"[CsvFileClient] 解析 CSV 文件: {csv_path}")
            df = pd.read_csv(csv_path, sep=';')
            
            print(f"[CsvFileClient] 解析成功，获取 {len(df)} 条记录")
            print(f"  - 列名: {list(df.columns)}")
            
            return df
            
        except grpc.RpcError as e:
            print(f"[CsvFileClient ERROR] gRPC 调用失败: {e.code()}: {e.details()}")
            return None
        except Exception as e:
            print(f"[CsvFileClient ERROR] 处理失败: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            # 5. 清理临时文件
            if cleanup_temp and os.path.exists(csv_path):
                try:
                    os.remove(csv_path)
                    print(f"[CsvFileClient] 已删除临时文件: {csv_path}")
                    
                    # 如果临时目录是自动创建的且为空，则删除目录
                    if temp_dir and os.path.isdir(temp_dir) and not os.listdir(temp_dir):
                        os.rmdir(temp_dir)
                        print(f"[CsvFileClient] 已删除临时目录: {temp_dir}")
                except Exception as e:
                    print(f"[CsvFileClient WARNING] 删除文件失败: {e}")
            
            # 6. 断开连接
            if not keep_file:
                self._disconnect()
    
    def close(self):
        """关闭客户端连接"""
        self._disconnect()


# 便捷函数
def load_dataset(dataset_name: str, 
                 server_address: str = None,
                 temp_dir: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    便捷函数：加载数据集
    
    Args:
        dataset_name: 数据集名称（CSV 文件名）
        server_address: WaterCsvFileService 的服务地址，如果为 None 则使用配置文件的默认地址
        temp_dir: 临时目录路径
        
    Returns:
        DataFrame: 解析后的数据，失败返回 None
    """
    client = CsvFileClient(server_address)
    try:
        return client.download_and_parse_csv(dataset_name, temp_dir)
    finally:
        client.close()
