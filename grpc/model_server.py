import grpc
import json
import os
import zipfile
import io
import shutil
from concurrent import futures
import water_pb2
import water_pb2_grpc


class WaterModelServiceServicer(water_pb2_grpc.WaterModelServiceServicer):
    def DownloadModel(self, request, context):
        """
        根据传来的JSON信息，找到对应模型文件夹，打包为zip文件返回
        """
        print("======================================")
        print("✅ Python 收到 Java gRPC 下载模型调用")
        print(f"✅ 任务编号: {request.task_no}")
        print(f"✅ 任务名称: {request.task_name}")
        print(f"✅ 模型路径: {request.model_json}")
        
        try:
            # 解析传来的 JSON 信息
            model_json = request.model_json
            if not model_json:
                return water_pb2.ModelDownloadResponse(
                    success=False,
                    message="model_json 为空",
                    file_content=b"",
                    file_name="",
                    model_type="",
                    model_version="",
                    r2=0.0,
                    mae=0.0,
                    rmse=0.0
                )
            
            # 解析 JSON 获取模型信息
            model_info = json.loads(model_json)
            model_path = model_info.get("modelPath", "")
            print("模型路径:", model_path)
            
            if not model_path or not os.path.exists(model_path):
                return water_pb2.ModelDownloadResponse(
                    success=False,
                    message=f"模型路径不存在: {model_path}",
                    file_content=b"",
                    file_name="",
                    model_type="",
                    model_version="",
                    r2=0.0,
                    mae=0.0,
                    rmse=0.0
                )
            
            # 检查是否为目录
            if not os.path.isdir(model_path):
                return water_pb2.ModelDownloadResponse(
                    success=False,
                    message=f"路径不是目录: {model_path}",
                    file_content=b"",
                    file_name="",
                    model_type="",
                    model_version="",
                    r2=0.0,
                    mae=0.0,
                    rmse=0.0
                )
            
            # 获取文件夹名称作为 zip 文件名
            folder_name = os.path.basename(model_path)
            zip_filename = f"{folder_name}.zip"
            
            # 创建内存中的 zip 文件
            zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # 1. 先将 model_json 保存为 JSON 文件并添加到压缩包
                json_filename = f"{folder_name}_info.json"
                zip_file.writestr(json_filename, model_json)
                print(f"📦 添加文件到压缩包: {json_filename}")
                
                # 2. 遍历文件夹中的所有文件
                for root, dirs, files in os.walk(model_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        # 计算在 zip 中的相对路径
                        arcname = os.path.relpath(file_path, os.path.dirname(model_path))
                        zip_file.write(file_path, arcname)
                        print(f"📦 添加文件到压缩包: {arcname}")
            
            # 获取 zip 文件的二进制内容
            zip_content = zip_buffer.getvalue()
            zip_buffer.close()
            
            print(f"✅ 压缩完成，文件大小: {len(zip_content)} bytes")
            
            # 从 config.pkl 中读取评估指标（如果存在）
            r2 = 0.0
            mae = 0.0
            rmse = 0.0
            
            config_path = os.path.join(model_path, "config.pkl")
            if os.path.exists(config_path):
                try:
                    import joblib
                    config = joblib.load(config_path)
                    metrics = config.get("metrics", {})
                    
                    # 计算所有目标的平均指标
                    if metrics:
                        r2_values = [m.get("r2", 0.0) for m in metrics.values()]
                        mae_values = [m.get("mae", 0.0) for m in metrics.values()]
                        rmse_values = [m.get("rmse", 0.0) for m in metrics.values()]
                        
                        r2 = sum(r2_values) / len(r2_values) if r2_values else 0.0
                        mae = sum(mae_values) / len(mae_values) if mae_values else 0.0
                        rmse = sum(rmse_values) / len(rmse_values) if rmse_values else 0.0
                        
                        print(f"📊 平均评估指标 - R²: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
                except Exception as e:
                    print(f"⚠️ 读取配置文件失败: {e}")
            
            # 构建响应
            response = water_pb2.ModelDownloadResponse(
                success=True,
                message=f"模型下载成功: {zip_filename}",
                file_content=zip_content,
                file_name=zip_filename,
                model_type="water_quality_prediction",
                model_version=model_info.get("model_name", ""),
                r2=r2,
                mae=mae,
                rmse=rmse
            )
            
            print("✅ 模型下载响应已发送")
            print("======================================")
            
            return response
            
        except json.JSONDecodeError as e:
            error_msg = f"JSON 解析失败: {str(e)}"
            print(f"❌ {error_msg}")
            return water_pb2.ModelDownloadResponse(
                success=False,
                message=error_msg,
                file_content=b"",
                file_name="",
                model_type="",
                model_version="",
                r2=0.0,
                mae=0.0,
                rmse=0.0
            )
        except Exception as e:
            error_msg = f"处理失败: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            return water_pb2.ModelDownloadResponse(
                success=False,
                message=error_msg,
                file_content=b"",
                file_name="",
                model_type="",
                model_version="",
                r2=0.0,
                mae=0.0,
                rmse=0.0
            )
    
    def UploadModel(self, request, context):
        """
        上传模型：接收 ZIP 文件，解压并只保存第一个文件夹下的所有文件
        """
        print("=" * 60)
        print("✅ Python 收到 Java gRPC 上传模型调用")
        print(f"✅ 任务编号: {request.task_no}")
        print(f"✅ 任务名称: {request.task_name}")
        print(f"✅ 模型名称: {request.model_name}")
        print(f"✅ 文件大小: {len(request.file_content)} bytes")
        print("=" * 60)
        
        try:
            # 1. 验证输入参数
            if not request.model_name:
                return water_pb2.ModelUploadResponse(
                    success=False,
                    message="model_name 不能为空",
                    model_path=""
                )
            
            if not request.file_content:
                return water_pb2.ModelUploadResponse(
                    success=False,
                    message="file_content 不能为空",
                    model_path=""
                )
            
            # 2. 创建目标文件夹
            models_base_dir = "models"
            target_folder = os.path.join(models_base_dir, request.model_name)
            
            # 如果文件夹已存在，先删除
            if os.path.exists(target_folder):
                shutil.rmtree(target_folder)
                print(f"⚠️ 已删除已存在的文件夹: {target_folder}")
            
            os.makedirs(target_folder, exist_ok=True)
            print(f"📁 创建目标文件夹: {target_folder}")
            
            # 3. 将 ZIP 文件内容保存到临时文件
            temp_zip_path = os.path.join(target_folder, "temp_upload.zip")
            with open(temp_zip_path, 'wb') as f:
                f.write(request.file_content)
            print(f"💾 ZIP 文件已保存: {temp_zip_path}")
            
            # 4. 解压 ZIP 文件，忽略 JSON 文件，直接保存其他文件
            with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                # 获取 ZIP 文件中的所有文件和文件夹
                zip_contents = zip_ref.namelist()
                print(f"📦 ZIP 文件包含 {len(zip_contents)} 个项目")
                
                extracted_count = 0
                skipped_json_count = 0
                
                for item in zip_contents:
                    # 跳过文件夹
                    if item.endswith('/') or item.endswith('\\'):
                        continue
                    
                    # 标准化路径（兼容 Windows）
                    normalized_item = item.replace('\\', '/')
                    
                    # 跳过 JSON 文件
                    if normalized_item.lower().endswith('.json'):
                        skipped_json_count += 1
                        print(f"   🚫 跳过 JSON: {item}")
                        continue
                    
                    # 提取文件名（去掉所有路径前缀）
                    filename = os.path.basename(normalized_item)
                    
                    if filename:  # 确保文件名不为空
                        # 创建目标文件路径（直接保存到目标文件夹根目录）
                        target_file_path = os.path.join(target_folder, filename)
                        
                        # 提取文件
                        with zip_ref.open(item) as source, open(target_file_path, 'wb') as target:
                            shutil.copyfileobj(source, target)
                        
                        extracted_count += 1
                        print(f"   ✅ 提取: {filename}")
                
                print(f"\n✅ 成功提取 {extracted_count} 个文件")
                print(f"🚫 跳过 {skipped_json_count} 个 JSON 文件")
            
            # 5. 删除临时 ZIP 文件
            if os.path.exists(temp_zip_path):
                os.remove(temp_zip_path)
                print(f"🗑️ 已删除临时文件: {temp_zip_path}")
            
            # 6. 验证解压结果
            print("\n📋 解压后的文件列表:")
            extracted_files = []
            for root, dirs, files in os.walk(target_folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, target_folder)
                    extracted_files.append(rel_path)
                    print(f"   📄 {rel_path}")
            
            if not extracted_files:
                print("\n❌ 错误: 解压后文件夹为空")
                return water_pb2.ModelUploadResponse(
                    success=False,
                    message="解压后文件夹为空",
                    model_path=target_folder
                )
            
            # 7. 读取评估指标（如果存在）
            r2 = request.r2 if request.r2 != 0 else 0.0
            mae = request.mae if request.mae != 0 else 0.0
            rmse = request.rmse if request.rmse != 0 else 0.0
            
            config_path = os.path.join(target_folder, "config.pkl")
            if os.path.exists(config_path):
                try:
                    import joblib
                    config = joblib.load(config_path)
                    metrics = config.get("metrics", {})
                    
                    # 如果 ZIP 中有 metrics，优先使用
                    if metrics:
                        r2_values = [m.get("r2", 0.0) for m in metrics.values()]
                        mae_values = [m.get("mae", 0.0) for m in metrics.values()]
                        rmse_values = [m.get("rmse", 0.0) for m in metrics.values()]
                        
                        r2 = sum(r2_values) / len(r2_values) if r2_values else r2
                        mae = sum(mae_values) / len(mae_values) if mae_values else mae
                        rmse = sum(rmse_values) / len(rmse_values) if rmse_values else rmse
                        
                        print(f"\n📊 从 config.pkl 读取指标 - R²: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
                except Exception as e:
                    print(f"\n⚠️ 读取配置文件失败: {e}")
            
            print(f"\n✅ 模型上传成功！")
            print(f"📁 模型路径: {target_folder}")
            print(f"📊 评估指标 - R²: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
            print("=" * 60)
            
            # 8. 返回响应
            return water_pb2.ModelUploadResponse(
                success=True,
                message=f"模型上传成功: {request.model_name}",
                model_path=target_folder
            )
            
        except zipfile.BadZipFile as e:
            error_msg = f"无效的 ZIP 文件: {str(e)}"
            print(f"\n❌ {error_msg}")
            return water_pb2.ModelUploadResponse(
                success=False,
                message=error_msg,
                model_path=""
            )
        except Exception as e:
            error_msg = f"上传失败: {str(e)}"
            print(f"\n❌ {error_msg}")
            import traceback
            traceback.print_exc()
            return water_pb2.ModelUploadResponse(
                success=False,
                message=error_msg,
                model_path=""
            )


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    water_pb2_grpc.add_WaterModelServiceServicer_to_server(
        WaterModelServiceServicer(), server
    )
    server.add_insecure_port("[::]:50053")
    print("✅ 模型管理 gRPC 服务启动：50053，等待 Java 调用...")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
