from concurrent import futures
import grpc

import water.proto.water_pb2 as water_pb2
import water.proto.water_pb2_grpc as water_pb2_grpc
from water.core import water_core
from water.utils.config import get_grpc_address, get_max_workers
from water.data.data_query import get_query_service


class WaterTrainServiceServicer(water_pb2_grpc.WaterTrainServiceServicer):
    def Train(self, request, context):
        print("======================================")
        print("收到 Java gRPC 调用!")
        
        # 获取训练参数
        predict_steps = request.predict_steps if request.predict_steps > 0 else 8  # 默认8小时
        train_steps = request.train_steps if request.train_steps > 0 else None
        model_type = request.model_type if request.model_type else "CatBoost"
        model_name = request.model_name if request.model_name else None
        
        print(f"模型类型: {model_type}")
        print(f"模型名称: {model_name}")
        print(f"预测步长: {predict_steps} hours")
        print(f"训练步长: {train_steps}")
        
        # 检查是否提供了数据标签和时间范围（新的数据查询方式）
        if request.data_tags and request.start_time and request.end_time:
            print(f"使用数据查询方式:")
            print(f"  - 数据标签: {list(request.data_tags)}")
            print(f"  - 目标变量: {request.target_variable}")
            print(f"  - 时间范围: {request.start_time} ~ {request.end_time}")
            print(f"  - 采样间隔: {request.sampling_interval}秒")
            
            try:
                # 使用数据查询服务获取数据
                query_service = get_query_service()
                df = query_service.query_data(
                    data_tags=list(request.data_tags),
                    target_variable=request.target_variable,
                    start_time=request.start_time,
                    end_time=request.end_time,
                    sampling_interval=request.sampling_interval if request.sampling_interval > 0 else 300,
                    source_type="database"  # 使用数据库查询
                )
                
                if df.empty:
                    return water_pb2.TrainResponse(
                        success=False,
                        message="未查询到训练数据",
                        model_name="",
                        model_path="",
                        scaler_path="",
                        config_path="",
                        metrics={}
                    )
                
                print(f"查询到 {len(df)} 条训练数据")
                
            except Exception as e:
                error_msg = f"数据查询失败: {str(e)}"
                print(f"[ERROR] {error_msg}")
                import traceback
                traceback.print_exc()
                
                return water_pb2.TrainResponse(
                    success=False,
                    message=error_msg,
                    model_name="",
                    model_path="",
                    scaler_path="",
                    config_path="",
                    metrics={}
                )
        else:
            # 兼容旧的方式：直接使用 data_list（如果 proto 中还有这个字段）
            print("WARNING: 使用旧的数据传入方式（data_list），建议升级到新的查询方式")
            # TODO: 如果 proto 中移除了 data_list，这里需要返回错误
            return water_pb2.TrainResponse(
                success=False,
                message="请使用新的数据查询方式：提供 data_tags、target_variable、start_time、end_time",
                model_name="",
                model_path="",
                scaler_path="",
                config_path="",
                metrics={}
            )
        
        print("开始训练模型...")
        print(f"DataFrame shape: {df.shape}")
        print(df.head())
        
        # 训练模型，传入目标变量和采样间隔
        model_result = water_core.train_model(
            df, 
            predict_steps=predict_steps, 
            model_type=model_type, 
            model_name=model_name,
            target_variable=request.target_variable if request.target_variable else None,
            sampling_interval=request.sampling_interval if request.sampling_interval > 0 else 300
        )

        print("训练完成! 返回 Java")
        
        # 构建 metrics map
        metrics_map = {}
        if "metrics" in model_result:
            for target_col, metric_data in model_result["metrics"].items():
                metrics_map[target_col] = water_pb2.ModelMetrics(
                    rmse=metric_data["rmse"],
                    mae=metric_data["mae"],
                    r2=metric_data["r2"]
                )
        
        return water_pb2.TrainResponse(
            success=True,
            message="训练完成",
            model_name=model_result["model_name"],
            model_path=model_result["model_path"],
            scaler_path=model_result["scaler_path"],
            config_path=model_result["config_path"],
            metrics=metrics_map
        )


def serve():
    max_workers = get_max_workers("train_service")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    water_pb2_grpc.add_WaterTrainServiceServicer_to_server(
        WaterTrainServiceServicer(), server
    )
    address = get_grpc_address("train_service")
    server.add_insecure_port(address)
    print(f"Python gRPC 服务启动：{address}，等待 Java 调用...")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
