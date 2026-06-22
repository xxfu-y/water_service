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
    
    # CSV 文件服务端口
    "csv_file_service": 50055,
}

# 外部 gRPC 服务配置
EXTERNAL_GRPC_SERVICES = {
    # 数据仓库历史数据服务
    "dw_history_service": {
        "host": "192.168.0.141",
        "port": 9090,
    },
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
    
    # CSV 文件服务工作线程数
    "csv_file_service": 5,
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


def get_external_service_address(service_name):
    """
    获取外部 gRPC 服务地址
    
    Args:
        service_name: 外部服务名称 (如 dw_history_service)
        
    Returns:
        str: 服务地址，如 "192.168.0.141:9090"
    """
    if service_name not in EXTERNAL_GRPC_SERVICES:
        raise ValueError(f"未知外部服务: {service_name}")
    
    service_config = EXTERNAL_GRPC_SERVICES[service_name]
    host = service_config["host"]
    port = service_config["port"]
    return f"{host}:{port}"


def get_external_service_host(service_name):
    """
    获取外部 gRPC 服务主机地址
    
    Args:
        service_name: 外部服务名称
        
    Returns:
        str: 主机地址，如 "192.168.0.141"
    """
    if service_name not in EXTERNAL_GRPC_SERVICES:
        raise ValueError(f"未知外部服务: {service_name}")
    
    return EXTERNAL_GRPC_SERVICES[service_name]["host"]


def get_external_service_port(service_name):
    """
    获取外部 gRPC 服务端口
    
    Args:
        service_name: 外部服务名称
        
    Returns:
        int: 端口号，如 9090
    """
    if service_name not in EXTERNAL_GRPC_SERVICES:
        raise ValueError(f"未知外部服务: {service_name}")
    
    return EXTERNAL_GRPC_SERVICES[service_name]["port"]
