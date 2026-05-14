import grpc
import pandas as pd
from concurrent import futures
import water_pb2
import water_pb2_grpc
import water_core

class PredictService(water_pb2_grpc.WaterPredictServiceServicer):
    def predict(self, request, context):
        rows = []
        for d in request.data_list:
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
                "bio2_1_blower_is_running": d.bio2_1_blower_is_running,
                "bio2_1_blower_pressure_meas": d.bio2_1_blower_pressure_meas,
                "bio2_2_blower_opening_meas": d.bio2_2_blower_opening_meas,
                "bio2_2_blower_is_running": d.bio2_2_blower_is_running,
                "bio2_2_blower_pressure_meas": d.bio2_2_blower_pressure_meas
            })
        df = pd.DataFrame(rows)
        res = water_core.predict_model(df)
        return water_pb2.PredictResponse(
            bio2_end_do_meas=res["bio2_end_do_meas"],
            outflow_cod=res["outflow_cod"],
            outflow_nh3=res["outflow_nh3"]
        )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    water_pb2_grpc.add_WaterPredictServiceServicer_to_server(PredictService(), server)
    server.add_insecure_port("[::]:50052")
    print("✅ 预测 gRPC 服务启动：50052")
    import time
    while True:
        time.sleep(3600)

if __name__ == "__main__":
    serve()