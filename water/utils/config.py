"""
gRPC 服务端口配置文件
统一管理所有 gRPC 服务的端口配置
"""

# gRPC 服务端口配置
GRPC_PORTS = {
    # 训练服务端口
    "train_service": 50051,
    
    # 预测服务端口
    "predict_service": 50052,
    
    # 模型管理服务端口（上传/下载）
    "model_service": 50053,
    
    # MPC 控制服务端口
    "mpc_service": 50054,
}

# 服务地址配置
GRPC_HOST = "[::]"  # 监听所有网络接口

# 并发工作线程配置
GRPC_WORKERS = {
    # 训练服务工作线程数
    "train_service": 10,
    
    # 预测服务工作线程数
    "predict_service": 16,
    
    # 模型管理工作线程数
    "model_service": 10,
    
    # MPC 控制服务工作线程数
    "mpc_service": 8,
}


def get_grpc_address(service_name):
    """
    获取 gRPC 服务地址
    
    Args:
        service_name: 服务名称 (train_service, predict_service, model_service, mpc_service)
        
    Returns:
        str: 完整的 gRPC 地址，如 "[::]:50051"
    """
    if service_name not in GRPC_PORTS:
        raise ValueError(f"未知服务: {service_name}")
    
    port = GRPC_PORTS[service_name]
    return f"{GRPC_HOST}:{port}"


def get_port(service_name):
    """
    获取指定服务的端口号
    
    Args:
        service_name: 服务名称
        
    Returns:
        int: 端口号
    """
    if service_name not in GRPC_PORTS:
        raise ValueError(f"未知服务: {service_name}")
    
    return GRPC_PORTS[service_name]


def get_max_workers(service_name):
    """
    获取指定服务的最大工作线程数
    
    Args:
        service_name: 服务名称
        
    Returns:
        int: 最大工作线程数
    """
    if service_name not in GRPC_WORKERS:
        raise ValueError(f"未知服务: {service_name}")
    
    return GRPC_WORKERS[service_name]
