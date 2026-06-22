#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
预测服务启动脚本
支持多种启动模式
"""

import sys
import os
import argparse

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.predict_server import serve


def main():
    parser = argparse.ArgumentParser(
        description='预测服务启动器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基础模式（仅响应客户端请求）
  python start_predict_service.py
  
  # 启用后台持续预测，每5秒预测一次
  python start_predict_service.py --continuous --interval 5
  
  # 自定义并发线程数
  python start_predict_service.py --workers 32
        """
    )
    
    parser.add_argument(
        '--continuous',
        action='store_true',
        help='启用后台持续预测模式'
    )
    
    parser.add_argument(
        '--interval',
        type=int,
        default=5,
        help='持续预测间隔（秒），默认5秒'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=16,
        help='并发工作线程数，默认16（至少支持双并发）'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("水质预测服务")
    print("=" * 70)
    print(f"启动模式: {'后台持续预测' if args.continuous else '基础模式'}")
    print(f"并发线程数: {args.workers}")
    if args.continuous:
        print(f"预测间隔: {args.interval}秒")
    print("=" * 70)
    print()
    
    # 注意：当前版本max_workers固定在16，如需自定义需要修改predict_server.py
    serve(
        enable_continuous=args.continuous,
        prediction_interval=args.interval
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n服务已停止")
        sys.exit(0)
    except Exception as e:
        print(f"\n服务启动失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
