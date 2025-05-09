#!/usr/bin/env python
"""
准备采样策略研究数据
从数据库获取SKU价格数据，并处理成适合采样实验的格式
"""
import os
import sys
import json
import logging
import argparse
from datetime import datetime

# 确保可以导入项目模块
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.data_fetcher import DataFetcher

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/sampling_data_preparation.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

def load_db_config():
    """加载数据库配置"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'database.json')
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='准备采样策略研究数据')
    parser.add_argument('--categories', type=str, nargs='+', help='要包含的类别列表，不指定则使用所有类别')
    parser.add_argument('--start_date', type=str, default='2022-01-01', help='数据开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2024-01-01', help='数据结束日期 (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default='data/processed/time_series_data.csv', help='输出文件路径')
    args = parser.parse_args()
    
    # 创建日志目录
    os.makedirs('logs', exist_ok=True)
    
    try:
        # 加载数据库配置
        db_config = load_db_config()
        
        # 创建数据获取器
        fetcher = DataFetcher(**db_config)
        
        # 生成数据集
        start_time = datetime.now()
        logger.info(f"开始数据准备 ({start_time})")
        
        dataset = fetcher.generate_sampling_dataset(
            categories=args.categories,
            start_date=args.start_date,
            end_date=args.end_date,
            output_path=args.output
        )
        
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"数据准备完成 ({end_time}), 耗时: {duration}")
        logger.info(f"数据已保存到: {args.output}")
        
    except Exception as e:
        logger.exception(f"数据准备过程中发生错误: {str(e)}")
        return 1
    finally:
        if 'fetcher' in locals():
            fetcher.close()
            
    return 0

if __name__ == "__main__":
    # 确保目录存在
    os.makedirs('data/processed', exist_ok=True)
    sys.exit(main()) 