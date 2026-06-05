import grpc
import pandas as pd
from concurrent import futures
import water_pb2
import water_pb2_grpc
import water_core


class WaterTrainServiceServicer(water_pb2_grpc.WaterTrainServiceServicer):
    def Train(self, request, context):
        print("======================================")
        print("✅ Python 收到 Java gRPC 调用！！！")
        print(f"✅ 条数：{len(request.data_list)}")
        rows = []
        for d in request.data_list:
            # 安全处理整数，防止 NaN
            def safe_int(val):
                try:
                    return int(val)
                except:
                    return 0

            rows.append({
                "id": d.id,
                "timestamp": d.timestamp,
                "inflow_rate2": d.inflow_rate2,
                "inflow_cod": d.inflow_cod,
                "inflow_nh3": d.inflow_nh3,
                "outflow_cod": d.outflow_cod,
                "outflow_nh3": d.outflow_nh3,
                "bio2_front_aeration_flow_meas": d.bio2_front_aeration_flow_meas,
                "bio2_front_valve_opening_meas": d.bio2_front_valve_opening_meas,
                "bio2_end_aeration_flow_meas": d.bio2_end_aeration_flow_meas,
                "bio2_end_valve_opening_meas": d.bio2_end_valve_opening_meas,
                "bio2_front_do_meas": d.bio2_front_do_meas,
                "bio2_end_do_meas": d.bio2_end_do_meas,
                "bio2_mlss_meas": d.bio2_mlss_meas,
                "bio2_aeration_pipe_pressure_meas": d.bio2_aeration_pipe_pressure_meas,
                "bio2_1_blower_opening_meas": d.bio2_1_blower_opening_meas,
                "bio2_1_blower_is_running": safe_int(d.bio2_1_blower_is_running),
                "bio2_1_blower_pressure_meas": d.bio2_1_blower_pressure_meas,
                "bio2_2_blower_opening_meas": d.bio2_2_blower_opening_meas,
                "bio2_2_blower_is_running": safe_int(d.bio2_2_blower_is_running),
                "bio2_2_blower_pressure_meas": d.bio2_2_blower_pressure_meas
            })
        print("DF",rows)
        df = pd.DataFrame(rows)
        print("✅ 开始训练模型...")
        
        # 获取 gRPC 传入的参数
        predict_steps = request.predict_steps if request.predict_steps > 0 else 8  # 默认8小时
        train_steps = request.train_steps if request.train_steps > 0 else None
        model_type = request.model_type if request.model_type else "CatBoost"
        
        print(f"📊 预测步长: {predict_steps} 小时")
        print(f"📊 训练步长: {train_steps}")
        print(f"📊 模型类型: {model_type}")
        
        model_result = water_core.train_model(df, predict_steps=predict_steps)

        print("✅ 训练完成！返回 Java")
        
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
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    water_pb2_grpc.add_WaterTrainServiceServicer_to_server(
        WaterTrainServiceServicer(), server
    )
    server.add_insecure_port("[::]:50051")
    print("✅ Python gRPC 服务启动：50051，等待 Java 调用...")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
