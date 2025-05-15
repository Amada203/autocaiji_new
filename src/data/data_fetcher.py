import pandas as pd
import logging
import os
import json
from impala.dbapi import connect
from impala.util import as_pandas
from typing import Dict, Any

class DataFetcher:
    def __init__(self):
        """初始化数据获取器"""
        self.config = self._load_config()
        self.logger = self._setup_logger()
        
    def _load_config(self) -> Dict[str, Any]:
        """加载Impala配置"""
        config_path = os.path.join(os.path.dirname(__file__), '../../config/database.json')
        try:
            with open(config_path) as f:
                config = json.load(f)
                impala_config = config.get('impala', {})
                
                # 关键配置校验
                required_keys = ['host', 'port', 'database']
                missing = [k for k in required_keys if k not in impala_config]
                if missing:
                    raise ValueError(f"缺少必要配置项: {missing}")
                    
                return impala_config
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {str(e)}")
            raise

    def _setup_logger(self):
        """设置日志记录器"""
        logger = logging.getLogger('DataFetcher')
        logger.setLevel(logging.INFO)
        
        # 确保日志目录存在
        os.makedirs('logs', exist_ok=True)
        
        # 文件日志
        file_handler = logging.FileHandler('logs/data_fetcher.log')
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        
        # 控制台日志
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            '%(levelname)s - %(message)s'
        ))
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        return logger

    def fetch_training_data(self) -> pd.DataFrame:
        """获取训练数据"""
        self.logger.info("开始从Impala获取训练数据")
        
        try:
            # 建立连接
            conn = connect(
                host=self.config['host'],
                port=self.config['port'],
                database=self.config['database'],
                user=self.config.get('user'),
                password=self.config.get('password'),
                auth_mechanism='PLAIN',
                timeout=30
            )
            
            # 执行查询
            query = """
            WITH active_skus AS (
                SELECT sku_id, COUNT(DISTINCT dt) AS cnt 
                FROM jd_daily_price 
                WHERE dt BETWEEN '2023-01-01' AND '2025-04-30'
                GROUP BY sku_id HAVING cnt >= 765
            )
            SELECT 
                b.sku_id,
                b.dt AS date,
                b.page_price,
                b.discount_price,
                CASE WHEN raw_promotion_detail IS NOT NULL THEN 1 ELSE 0 END AS is_promotion
            FROM jd_daily_price b
            JOIN active_skus a ON a.sku_id = b.sku_id
            """
            
            cursor = conn.cursor()
            cursor.execute(query)
            df = as_pandas(cursor)
            
            self.logger.info(f"成功获取 {len(df)} 条记录")
            return df
            
        except Exception as e:
            self.logger.error(f"获取数据失败: {str(e)}")
            raise
        finally:
            if 'conn' in locals():
                conn.close()

if __name__ == "__main__":
    # 测试数据获取
    fetcher = DataFetcher()
    try:
        data = fetcher.fetch_training_data()
        print(f"获取数据成功，记录数: {len(data)}")
    except Exception as e:
        print(f"测试失败: {str(e)}")