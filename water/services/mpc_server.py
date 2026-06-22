"""
MPC 控制服务 - gRPC 服务端
支持云-边协同架构、多实例管理、双模型运行
"""

import json
from concurrent import futures
import grpc

import water.proto.water_pb2 as water_pb2
import water.proto.water_pb2_grpc as water_pb2_grpc
from water.data.mpc_controller import MpcController
from water.utils.config import get_grpc_address, get_max_workers
from water.data.data_query import get_query_service


class WaterMpcServiceServicer(water_pb2_grpc.WaterMpcServiceServicer):
    """MPC 控制服务实现 - 支持多实例管理"""
    
    def __init__(self):
        # 存储控制器实例：key = instance_id
        self.controllers = {}
        print("MPC 服务初始化完成（支持多实例、双模型）")
    
    def Control(self, request, context):
        """
        执行 MPC 控制
        
        Args:
            request: MpcControlRequest
            context: gRPC context
            
        Returns:
            MpcControlResponse
        """
        try:
            print("=" * 60)
            print("收到 MPC 控制请求")
            
            # 1. 解析配置
            config_proto = request.config
            config_dict = self._proto_config_to_dict(config_proto)
            
            instance_id = config_dict.get("instanceId", "default")
            controller_name = config_dict.get("controllerName", f"mpc-{instance_id}")
            use_shadow = config_dict.get("useShadowModel", False)
            
            print(f"实例 ID: {instance_id}")
            print(f"控制器名称: {controller_name}")
            print(f"使用影子模型: {use_shadow}")
            
            # 2. 获取或创建控制器（按实例 ID）
            if instance_id not in self.controllers:
                print(f"创建新实例控制器: {instance_id}")
                self.controllers[instance_id] = MpcController(instance_id, config_dict)
            
            controller = self.controllers[instance_id]
            
            # 3. 加载模型（如果配置了模型路径）
            formal_model_path = config_dict.get("formalModelPath")
            shadow_model_path = config_dict.get("shadowModelPath")
            
            if formal_model_path and controller.model_manager.formal_model is None:
                controller.model_manager.load_formal_model(formal_model_path)
            
            if shadow_model_path and controller.model_manager.shadow_model is None:
                controller.model_manager.load_shadow_model(shadow_model_path)
            
            # 4. 解析传感器数据
            sensor_data = []
            
            # 检查是否提供了数据查询参数（新的方式）
            if hasattr(request, 'data_query') and request.data_query:
                print(f"使用数据查询方式获取传感器数据:")
                query = request.data_query
                print(f"  - 数据标签: {list(query.data_tags)}")
                print(f"  - 时间范围: {query.start_time} ~ {query.end_time}")
                print(f"  - 采样间隔: {query.sampling_interval}秒")
                
                try:
                    # 使用数据查询服务获取数据
                    query_service = get_query_service()
                    df = query_service.query_data(
                        data_tags=list(query.data_tags),
                        target_variable=query.target_variable if query.target_variable else "",
                        start_time=query.start_time,
                        end_time=query.end_time,
                        sampling_interval=query.sampling_interval if query.sampling_interval > 0 else 60,
                        source_type="database"  # 使用数据库查询
                    )
                    
                    if df.empty:
                        return water_pb2.MpcControlResponse(
                            success=False,
                            message="未查询到传感器数据",
                            controller_name=controller_name,
                            instance_id=instance_id,
                            mv_value=0.0,
                            predicted_cv=0.0,
                            control_status="ERROR",
                            health_score=0.0,
                            model_used="ERROR",
                            timestamp=""
                        )
                    
                    # 将 DataFrame 转换为 SensorData 格式
                    from water.data.data_adapter import DataAdapter
                    adapter = DataAdapter()
                    sensor_data = adapter.dataframe_to_sensor_data(df)
                    print(f"查询到 {len(sensor_data)} 条传感器数据")
                    
                except Exception as e:
                    error_msg = f"数据查询失败: {str(e)}"
                    print(f"[ERROR] {error_msg}")
                    import traceback
                    traceback.print_exc()
                    
                    return water_pb2.MpcControlResponse(
                        success=False,
                        message=error_msg,
                        controller_name=controller_name,
                        instance_id=instance_id,
                        mv_value=0.0,
                        predicted_cv=0.0,
                        control_status="ERROR",
                        health_score=0.0,
                        model_used="ERROR",
                        timestamp=""
                    )
            elif request.sensor_data:
                # 兼容旧的方式：直接使用 sensor_data
                print(f"使用直接传入的传感器数据:")
                for data in request.sensor_data:
                    sensor_data.append({
                        "timestamp": data.timestamp,
                        "values": dict(data.values)
                    })
                print(f"接收到 {len(sensor_data)} 条传感器数据")
            else:
                return water_pb2.MpcControlResponse(
                    success=False,
                    message="请提供传感器数据：使用 sensor_data 或 data_query",
                    controller_name=controller_name,
                    instance_id=instance_id,
                    mv_value=0.0,
                    predicted_cv=0.0,
                    control_status="ERROR",
                    health_score=0.0,
                    model_used="ERROR",
                    timestamp=""
                )
            
            # 5. 执行控制计算（支持双模型）
            result = controller.compute_control(sensor_data, use_shadow=use_shadow)
            
            print(f"控制结果:")
            print(f"  - MV 值: {result['mv_value']:.4f}")
            print(f"  - 预测 CV: {result['predicted_cv']:.4f}")
            print(f"  - 状态: {result['control_status']}")
            print(f"  - 健康度: {result['health_score']:.4f}")
            print(f"  - 使用模型: {result['model_used']}")
            
            # 6. 构建响应
            response = water_pb2.MpcControlResponse(
                success=result["success"],
                message=result["message"],
                controller_name=result["controller_name"],
                instance_id=result.get("instance_id", instance_id),
                mv_value=result["mv_value"],
                predicted_cv=result["predicted_cv"],
                control_status=result["control_status"],
                health_score=result["health_score"],
                model_used=result.get("model_used", "UNKNOWN"),
                timestamp=result["timestamp"]
            )
            
            print("=" * 60)
            return response
            
        except Exception as e:
            error_msg = f"MPC 控制失败: {str(e)}"
            print(f"错误: {error_msg}")
            import traceback
            traceback.print_exc()
            
            return water_pb2.MpcControlResponse(
                success=False,
                message=error_msg,
                controller_name=request.config.controller_name if request.config else "unknown",
                instance_id="",
                mv_value=0.0,
                predicted_cv=0.0,
                control_status="ERROR",
                health_score=0.0,
                model_used="ERROR",
                timestamp=""
            )
    
    def UpdateConfig(self, request, context):
        """
        更新控制器配置
        
        Args:
            request: MpcControllerConfig
            context: gRPC context
            
        Returns:
            MpcControlResponse
        """
        try:
            print("=" * 60)
            print("收到配置更新请求")
            
            config_dict = self._proto_config_to_dict(request)
            controller_name = config_dict.get("controllerName", "unknown")
            
            # 更新或创建控制器
            if controller_name in self.controllers:
                self.controllers[controller_name].update_config(config_dict)
                message = f"控制器 {controller_name} 配置已更新"
            else:
                self.controllers[controller_name] = MpcController(config_dict)
                message = f"创建新控制器 {controller_name}"
            
            print(message)
            print("=" * 60)
            
            return water_pb2.MpcControlResponse(
                success=True,
                message=message,
                controller_name=controller_name,
                mv_value=0.0,
                predicted_cv=0.0,
                control_status="NORMAL",
                health_score=1.0,
                timestamp=""
            )
            
        except Exception as e:
            error_msg = f"配置更新失败: {str(e)}"
            print(f"错误: {error_msg}")
            
            return water_pb2.MpcControlResponse(
                success=False,
                message=error_msg,
                controller_name=request.controller_name if request else "unknown",
                mv_value=0.0,
                predicted_cv=0.0,
                control_status="ERROR",
                health_score=0.0,
                timestamp=""
            )
    
    def GetStatus(self, request, context):
        """
        获取控制器状态
        
        Args:
            request: MpcControllerConfig
            context: gRPC context
            
        Returns:
            MpcControlResponse
        """
        try:
            controller_name = request.controller_name
            
            if controller_name in self.controllers:
                status = self.controllers[controller_name].get_status()
                
                return water_pb2.MpcControlResponse(
                    success=True,
                    message=f"控制器 {controller_name} 状态正常",
                    controller_name=controller_name,
                    mv_value=status.get("last_mv", 0.0),
                    predicted_cv=0.0,
                    control_status=status.get("status", "UNKNOWN"),
                    health_score=1.0,
                    timestamp=status.get("timestamp", "")
                )
            else:
                return water_pb2.MpcControlResponse(
                    success=False,
                    message=f"控制器 {controller_name} 不存在",
                    controller_name=controller_name,
                    mv_value=0.0,
                    predicted_cv=0.0,
                    control_status="ERROR",
                    health_score=0.0,
                    timestamp=""
                )
                
        except Exception as e:
            error_msg = f"获取状态失败: {str(e)}"
            print(f"错误: {error_msg}")
            
            return water_pb2.MpcControlResponse(
                success=False,
                message=error_msg,
                controller_name=request.controller_name if request else "unknown",
                mv_value=0.0,
                predicted_cv=0.0,
                control_status="ERROR",
                health_score=0.0,
                timestamp=""
            )
    
    def _proto_config_to_dict(self, config_proto) -> dict:
        """
        将 proto 配置转换为字典
        
        Args:
            config_proto: MpcControllerConfig proto 对象
            
        Returns:
            dict: 配置字典
        """
        # 转换特征绑定
        feature_bindings = []
        for binding in config_proto.feature_bindings:
            feature_bindings.append({
                "type": binding.type,
                "featureName": binding.feature_name,
                "physicalTag": binding.physical_tag
            })
        
        # 转换自动重训练配置
        auto_retrain = None
        if config_proto.HasField("auto_retrain"):
            auto_retrain = {
                "metricType": config_proto.auto_retrain.metric_type,
                "operator": config_proto.auto_retrain.operator,
                "threshold": config_proto.auto_retrain.threshold,
                "consecutiveCount": config_proto.auto_retrain.consecutive_count
            }
        
        # 转换自动切换配置
        auto_switch = None
        if config_proto.HasField("auto_switch"):
            auto_switch = {
                "observationPeriod": config_proto.auto_switch.observation_period,
                "minValidSamples": config_proto.auto_switch.min_valid_samples,
                "rmseImprovementThreshold": config_proto.auto_switch.rmse_improvement_threshold,
                "consecutiveSatisfyCount": config_proto.auto_switch.consecutive_satisfy_count
            }
        
        # 转换健康度阈值配置
        health_threshold = None
        if config_proto.HasField("health_threshold"):
            health_threshold = {
                "goodThreshold": config_proto.health_threshold.good_threshold,
                "normalThreshold": config_proto.health_threshold.normal_threshold
            }
        
        # 构建完整配置字典
        config_dict = {
            "controllerName": config_proto.controller_name,
            "instanceId": config_proto.instance_id,
            "instanceName": config_proto.instance_name,
            "status": config_proto.status,
            "predictDuration": config_proto.predict_duration,
            "featureBindings": feature_bindings,
            "mpcEnabled": config_proto.mpc_enabled,
            "cvTag": config_proto.cv_tag,
            "mvTag": config_proto.mv_tag,
            "samplingTime": config_proto.sampling_time,
            "predictionHorizon": config_proto.prediction_horizon,
            "controlHorizon": config_proto.control_horizon,
            "targetSP": config_proto.target_sp,
            "deadbandLower": config_proto.deadband_lower,
            "deadbandUpper": config_proto.deadband_upper,
            "safetyLower": config_proto.safety_lower,
            "safetyUpper": config_proto.safety_upper,
            "mvPhysicalLower": config_proto.mv_physical_lower,
            "mvPhysicalUpper": config_proto.mv_physical_upper,
            "mvMinChange": config_proto.mv_min_change,
            "mvMaxChange": config_proto.mv_max_change,
            "formalModelPath": config_proto.formal_model_path,
            "shadowModelPath": config_proto.shadow_model_path,
            "useShadowModel": config_proto.use_shadow_model,
            "autoRetrainCondition": auto_retrain,
            "autoSwitchToFormal": config_proto.auto_switch_to_formal,
            "autoSwitchConfig": auto_switch,
            "healthThresholdConfig": health_threshold
        }
        
        return config_dict


def serve():
    """启动 MPC 控制服务"""
    max_workers = get_max_workers("mpc_service")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    water_pb2_grpc.add_WaterMpcServiceServicer_to_server(
        WaterMpcServiceServicer(), server
    )
    address = get_grpc_address("mpc_service")
    server.add_insecure_port(address)
    print(f"MPC 控制 gRPC 服务启动：{address}，等待调用...")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
