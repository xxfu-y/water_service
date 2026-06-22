# water_service

## 🚀 快速启动

### Windows 系统

**方式一：交互式启动（推荐）**
```bash
# 双击运行或命令行执行
scripts\start.bat
```
然后按提示选择要启动的服务：
- 输入 `1` - 启动训练服务 (50051)
- 输入 `2` - 启动预测服务 (50052)
- 输入 `3` - 启动模型管理服务 (50053)
- 输入 `4` - 启动所有服务

**方式二：直接启动单个服务**
```bash
# 启动训练服务
scripts\start_train.bat

# 启动预测服务
scripts\start_predict.bat

# 启动模型管理服务
scripts\start_model.bat
```

### Linux/Mac 系统

**方式一：交互式启动（推荐）**
```bash
# 添加执行权限
chmod +x scripts/*.sh

# 运行启动脚本
./scripts/start.sh
```

**方式二：直接启动单个服务**
```bash
# 添加执行权限
chmod +x scripts/*.sh

# 启动训练服务
./scripts/start_train.sh

# 启动预测服务
./scripts/start_predict.sh

# 启动模型管理服务
./scripts/start_model.sh
```

### 停止服务

```bash
# Windows
scripts\stop.bat

# Linux/Mac
chmod +x scripts/stop.sh
./scripts/stop.sh
```

### 手动启动

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 生成 gRPC 代码（首次运行需要）
cd grpc
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. water.proto
cd ..

# 3. 启动服务
cd grpc
python train_server.py      # 训练服务
python predict_server.py    # 预测服务
python model_server.py      # 模型管理服务
```

---

## 📝 配置文件位置
`grpc/config.py`

## 端口配置

所有 gRPC 服务的端口都在 `config.py` 中统一管理：

| 服务名称 | 端口号 | 说明 |
|---------|--------|------|
| train_service | 50051 | 模型训练服务 |
| predict_service | 50052 | 水质预测服务 |
| model_service | 50053 | 模型管理服务（上传/下载） |
| mpc_service | 50054 | MPC 控制服务 |

## 工作线程配置

每个服务的并发工作线程数也在配置文件中定义：

| 服务名称 | 工作线程数 | 说明 |
|---------|-----------|------|
| train_service | 10 | 训练服务工作线程 |
| predict_service | 16 | 预测服务工作线程（支持高并发） |
| model_service | 10 | 模型管理工作线程 |

## 修改端口

如需修改端口，只需编辑 `config.py` 文件中的 `GRPC_PORTS` 字典：

```python
GRPC_PORTS = {
    "train_service": 50051,      # 修改此处的端口号
    "predict_service": 50052,
    "model_service": 50053,
}
```

## 修改工作线程数

如需调整并发性能，编辑 `GRPC_WORKERS` 字典：

```python
GRPC_WORKERS = {
    "train_service": 10,         # 修改此处的工作线程数
    "predict_service": 16,
    "model_service": 10,
}
```

## 使用配置

在所有服务文件中，通过以下方式获取配置：

```python
from config import get_grpc_address, get_max_workers

# 获取服务地址
address = get_grpc_address("train_service")  # 返回 "[::]:50051"

# 获取端口号
port = get_port("train_service")  # 返回 50051

# 获取最大工作线程数
max_workers = get_max_workers("train_service")  # 返回 10
```

## 注意事项

1. **端口冲突**：确保配置的端口未被其他程序占用
2. **防火墙设置**：如需远程访问，请在防火墙中开放相应端口
3. **Java 客户端**：修改端口后，需要同步更新 Java 客户端的端口配置
4. **统一调度**：所有端口必须在此配置文件中定义，禁止硬编码
