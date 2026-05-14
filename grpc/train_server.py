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
        water_core.train_model(df)

        print("✅ 训练完成！返回 Java")
        return water_pb2.TrainResponse(success=True, message="训练完成，模型已保存")


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