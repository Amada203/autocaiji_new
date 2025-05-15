import mysql.connector
import pandas as pd
import logging
import os
import sys
from datetime import datetime
from typing import List, Dict, Any

# 初始化日志目录
os.makedirs("logs", exist_ok=True)

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/mysql_writer.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class MySQLWriter:
    def __init__(self):
        """初始化MySQL写入器"""
        self.db_config = self._load_config()
        self.connection = None
        self.batch_size = 1000  # 批量写入大小

    def _load_config(self) -> Dict[str, Any]:
        """加载数据库配置（当前保持明文，待优化为环境变量）"""
        return {
            'host': 'localhost',
            'port': 13306,
            'user': 'root',
            'password': 'mypassword123',  # TODO: 改为环境变量
            'database': 'price_prediction',
            'auth_plugin': 'mysql_native_password'
        }

    def connect(self) -> bool:
        """连接到MySQL数据库"""
        try:
            self.connection = mysql.connector.connect(
                **self.db_config,
                connect_timeout=10,
                autocommit=False
            )
            logger.info("成功连接到MySQL数据库")
            return True
        except mysql.connector.Error as err:
            logger.error(f"连接MySQL失败 (错误 {err.errno}): {err.msg}")
            return False

    def write_predictions(self, predictions_df: pd.DataFrame) -> bool:
        """批量写入预测结果"""
        if not self.connect():
            return False

        try:
            cursor = self.connection.cursor()
            
            # 准备批量插入语句
            query = """
            INSERT INTO sku_predictions 
            (sku_id, sku_name, current_price, predicted_prob)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            sku_name = VALUES(sku_name),
            current_price = VALUES(current_price),
            predicted_prob = VALUES(predicted_prob),
            last_updated = CURRENT_TIMESTAMP
            """
            
            # 分批处理大数据集
            records = [
                (row['sku_id'], row['sku_name'], row['current_price'], row['predicted_prob'])
                for _, row in predictions_df.iterrows()
            ]
            
            for i in range(0, len(records), self.batch_size):
                batch = records[i:i + self.batch_size]
                cursor.executemany(query, batch)
                self.connection.commit()
                logger.info(f"已写入 {len(batch)} 条记录 (总计: {min(i + len(batch), len(records))}/{len(records)})")
            
            return True
            
        except Exception as e:
            logger.error(f"写入预测结果失败: {str(e)}")
            if self.connection:
                self.connection.rollback()
            return False
        finally:
            if self.connection and self.connection.is_connected():
                cursor.close()
                self.connection.close()

    def write_history(self, history_df: pd.DataFrame) -> bool:
        """批量写入历史数据"""
        if not self.connect():
            return False

        try:
            cursor = self.connection.cursor()
            query = """
            INSERT IGNORE INTO sku_history 
            (sku_id, date, price, is_promotion)
            VALUES (%s, %s, %s, %s)
            """
            
            records = [
                (row['sku_id'], row['date'], row['price'], row['is_promotion'])
                for _, row in history_df.iterrows()
            ]
            
            for i in range(0, len(records), self.batch_size):
                batch = records[i:i + self.batch_size]
                cursor.executemany(query, batch)
                self.connection.commit()
                logger.info(f"已写入 {len(batch)} 条历史记录 (总计: {min(i + len(batch), len(records))}/{len(records)})")
            
            return True
            
        except Exception as e:
            logger.error(f"写入历史数据失败: {str(e)}")
            if self.connection:
                self.connection.rollback()
            return False
        finally:
            if self.connection and self.connection.is_connected():
                cursor.close()
                self.connection.close()

if __name__ == "__main__":
    # 测试用例
    writer = MySQLWriter()
    
    # 测试数据
    test_pred = pd.DataFrame({
        'sku_id': ['test_001', 'test_002'],
        'sku_name': ['测试商品A', '测试商品B'],
        'current_price': [99.9, 199.9],
        'predicted_prob': [0.85, 0.72]
    })
    
    test_hist = pd.DataFrame({
        'sku_id': ['test_001', 'test_002'],
        'date': ['2023-01-01', '2023-01-02'],
        'price': [89.9, 179.9],
        'is_promotion': [True, False]
    })
    
    # 执行测试
    writer.write_predictions(test_pred)
    writer.write_history(test_hist)