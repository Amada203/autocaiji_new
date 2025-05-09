import sys
import os
import logging
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_fetcher import DataFetcher
from datetime import datetime, timedelta

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_db_config():
    """加载数据库配置"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'database.json')
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    # 加载数据库配置
    impala_config = load_db_config()
    
    # 创建数据获取器
    fetcher = DataFetcher(**impala_config)
    
    try:
        # 设置日期范围
        start_date = '2023-01-01'  # 开始日期
        end_date = '2025-04-01'    # 结束日期
        base_date = '2023-01-01'   # 基准日期，用于获取商品排名
        
        # 获取清洁用品价格数据
        logger.info("开始获取清洁用品价格数据...")
        df = fetcher.fetch_cleaning_data(
            start_date=start_date,
            end_date=end_date,
            base_date=base_date,
            max_items_per_category=300  # 每个类别最多商品数
        )
        
        # 保存数据
        output_dir = 'data/raw'
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'cleaning_data.csv')
        df.to_csv(output_file, index=False)
        logger.info(f"数据已保存到: {output_file}")
        
        # 输出数据统计
        logger.info("\n数据统计信息:")
        logger.info(f"数据时间范围: {df['record_date'].min()} 至 {df['record_date'].max()}")
        logger.info(f"SKU数量: {df['sku_id'].nunique()}")
        logger.info(f"总记录数: {len(df)}")
        logger.info(f"价格变动次数: {len(df[df['price'] != df['price'].shift(1)])}")
        
        # 输出价格分布统计
        logger.info("\n价格分布:")
        price_stats = df.groupby('category_name').agg({
            'price': ['count', 'mean', 'std', 'min', 'max'],
            'is_promotion': 'mean'
        })
        logger.info("\n" + str(price_stats))
        
        # 输出每个类别的SKU数量
        logger.info("\n每个类别的SKU数量:")
        sku_counts = df.groupby('category_name')['sku_id'].nunique()
        logger.info("\n" + str(sku_counts))
        
        # 输出每个类别的记录数
        logger.info("\n每个类别的记录数:")
        record_counts = df.groupby('category_name').size()
        logger.info("\n" + str(record_counts))
        
    finally:
        # 关闭数据库连接
        fetcher.close()

if __name__ == "__main__":
    main() 