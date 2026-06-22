import pandas as pd
from concurrent import futures
import grpc

import water.proto.water_pb2 as water_pb2
import water.proto.water_pb2_grpc as water_pb2_grpc
from water.core import water_core
import time
import threading
from queue import Queue, Empty
from water.utils.config import get_grpc_address, get_max_workers
from water.data.data_adapter import DataAdapter

class PredictService(water_pb2_grpc.WaterPredictServiceServicer):
    def __init__(self):
        # 数据队列，用于持续接收数据
        self.data_queue = Queue()
        # 预测结果缓存
        self.latest_predictions = {}
        # 运行状态
        self.running = False
        # 后台预测线程
        self.prediction_thread = None
    
    def predict(self, request, context):
        """单次预测接口 - 支持并发调用"""
        # 使用数据适配器转换数据
        adapter = DataAdapter()
        
        # 将 proto 数据转换为字典列表
        sensor_data_list = []
        for d in request.data_list:
            sensor_data_list.append({
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
        
        # 使用适配器转换为 DataFrame
        df = adapter.water_sensor_data_to_dataframe(sensor_data_list)
        
        if df.empty:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("No valid data provided")
            return water_pb2.PredictResponse(
                bio2_end_do_meas=0.0,
                outflow_cod=0.0,
                outflow_nh3=0.0
            )
        
        res = water_core.predict_model(df)
        return water_pb2.PredictResponse(
            bio2_end_do_meas=res["bio2_end_do_meas"],
            outflow_cod=res["outflow_cod"],
            outflow_nh3=res["outflow_nh3"]
        )
    
    def continuous_predict(self, request_iterator, context):
        """持续预测接口 - 客户端流式上传数据，服务端流式返回预测结果"""
        print("启动持续预测服务")
        adapter = DataAdapter()
        
        for request in request_iterator:
            try:
                # 将 proto 数据转换为字典列表
                sensor_data_list = []
                for d in request.data_list:
                    sensor_data_list.append({
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
                
                if not sensor_data_list:
                    continue
                
                # 使用适配器转换为 DataFrame
                df = adapter.water_sensor_data_to_dataframe(sensor_data_list)
                
                if df.empty:
                    continue
                
                res = water_core.predict_model(df)
                
                # 更新最新预测结果
                self.latest_predictions = res
                
                # 返回预测结果
                yield water_pb2.PredictResponse(
                    bio2_end_do_meas=res["bio2_end_do_meas"],
                    outflow_cod=res["outflow_cod"],
                    outflow_nh3=res["outflow_nh3"]
                )
                
            except Exception as e:
                print(f"预测错误: {e}")
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(str(e))
                return
    
    def start_continuous_prediction(self, data_source_func, interval_seconds=5):
        """启动后台持续预测
        
        Args:
            data_source_func: 数据获取函数，返回DataFrame或None
            interval_seconds: 预测间隔（秒）
        """
        self.running = True
        
        def prediction_loop():
            print(f"后台持续预测已启动，间隔: {interval_seconds}秒")
            while self.running:
                try:
                    # 获取数据
                    df = data_source_func()
                    if df is not None and not df.empty:
                        # 执行预测
                        res = water_core.predict_model(df)
                        self.latest_predictions = res
                        print(f"预测完成 - DO: {res['bio2_end_do_meas']:.4f}, COD: {res['outflow_cod']:.4f}, NH3: {res['outflow_nh3']:.4f}")
                    else:
                        print("无新数据")
                except Exception as e:
                    print(f"后台预测错误: {e}")
                
                # 等待下一个周期
                time.sleep(interval_seconds)
        
        self.prediction_thread = threading.Thread(target=prediction_loop, daemon=True)
        self.prediction_thread.start()
    
    def stop_continuous_prediction(self):
        """停止后台持续预测"""
        self.running = False
        if self.prediction_thread:
            self.prediction_thread.join(timeout=10)
            print("后台持续预测已停止")
    
    def get_latest_predictions(self):
        """获取最新预测结果"""
        return self.latest_predictions.copy() if self.latest_predictions else None

def serve(enable_continuous=False, prediction_interval=5):
    """启动预测服务
    
    Args:
        enable_continuous: 是否启用后台持续预测
        prediction_interval: 持续预测间隔（秒）
    """
    # 从配置文件获取并发工作线程数
    max_workers = get_max_workers("predict_service")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    
    predict_service = PredictService()
    water_pb2_grpc.add_WaterPredictServiceServicer_to_server(predict_service, server)
    address = get_grpc_address("predict_service")
    server.add_insecure_port(address)
    
    print(f"预测 gRPC 服务启动：{address}")
    print(f"并发工作线程数：{max_workers}")
    
    # 如果启用持续预测，需要提供一个数据获取函数
    if enable_continuous:
        # 示例：从数据库或API获取数据的函数
        def fetch_data_from_source():
            """这里替换为你的实际数据获取逻辑"""
            # 示例：从数据库、消息队列、或其他数据源获取数据
            # return your_data_fetch_function()
            return None  # 暂时返回None，需要根据实际情况实现
        
        predict_service.start_continuous_prediction(
            data_source_func=fetch_data_from_source,
            interval_seconds=prediction_interval
        )
        print(f"后台持续预测已启用，间隔: {prediction_interval}秒")
    
    try:
        server.start()
        print("服务正在运行...\n")
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\n收到停止信号")
        if enable_continuous:
            predict_service.stop_continuous_prediction()
        server.stop(0)
        print("服务已停止")

if __name__ == "__main__":
    serve()