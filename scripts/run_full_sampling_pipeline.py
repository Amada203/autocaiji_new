#!/usr/bin/env python
"""
商品价格智能采样策略完整测试流程
整合数据获取、SKU聚类和采样调度的全过程
"""
import os
import sys
import argparse
import pandas as pd
from datetime import datetime

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='运行商品价格智能采样策略完整测试流程')
    
    # 参数定义
    parser.add_argument('--start_date', type=str, default='2023-01-01', 
                      help='开始日期(YYYY-MM-DD), 默认:2023-01-01')
    parser.add_argument('--end_date', type=str, default='2025-04-30',
                      help='结束日期(YYYY-MM-DD), 默认:2025-04-30')
    parser.add_argument('--split_date', type=str, default='2024-12-31',
                      help='训练/测试分割日期(YYYY-MM-DD), 默认:2024-12-31')
    parser.add_argument('--capture_rate', type=float, default=0.95,
                      help='目标捕获率(0.8-0.99), 默认:0.95')
    parser.add_argument('--output_dir', type=str, default='output',
                      help='输出目录路径, 默认:output')
    parser.add_argument('--forecast_days', type=int, default=7,
                      help='预测天数(1-365), 默认:7')
    
    args = parser.parse_args()
    
    # 参数验证
    try:
        # 检查日期格式
        start_date = pd.to_datetime(args.start_date)
        end_date = pd.to_datetime(args.end_date)
        split_date = pd.to_datetime(args.split_date)
        
        # 验证日期顺序
        if start_date >= end_date:
            raise ValueError("开始日期必须早于结束日期")
        if split_date <= start_date or split_date >= end_date:
            raise ValueError("分割日期必须在开始和结束日期之间")
            
        # 验证数值范围
        if not 0.8 <= args.capture_rate <= 0.99:
            raise ValueError("捕获率必须在0.8到0.99之间")
        if args.forecast_days < 1 or args.forecast_days > 365:
            raise ValueError("预测天数必须在1到365之间")
            
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        
    except ValueError as e:
        print(f"参数错误: {str(e)}", file=sys.stderr)
        return 1
    except OSError as e:
        print(f"目录创建失败: {str(e)}", file=sys.stderr)
        return 1
    
    print("参数验证通过，开始执行主流程...")
    # 主流程继续...
    
    return 0

if __name__ == '__main__':
    sys.exit(main())