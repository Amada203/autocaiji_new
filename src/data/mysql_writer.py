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
        """连接到MySQL数据库并确保表存在"""
        try:
            self.connection = mysql.connector.connect(
                **self.db_config,
                connect_timeout=10,
                autocommit=False
            )
            logger.info("成功连接到MySQL数据库")
            
            # 检查并创建必要的表
            cursor = self.connection.cursor()
            
            # 强制重建预测结果表
            cursor.execute("DROP TABLE IF EXISTS sku_predictions")
            cursor.execute("""
                CREATE TABLE sku_predictions (
                    sku VARCHAR(50) NOT NULL PRIMARY KEY,
                    date DATETIME,
                    price DECIMAL(10,2),
                    confidence FLOAT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            logger.info("已重建sku_predictions表")
            
            # 强制重建历史数据表
            cursor.execute("DROP TABLE IF EXISTS sku_history")
            cursor.execute("""
                CREATE TABLE sku_history (
                    sku_id VARCHAR(50) NOT NULL,
                    date DATE NOT NULL,
                    price DECIMAL(10,2) NOT NULL,
                    is_promotion BOOLEAN DEFAULT FALSE,
                    PRIMARY KEY (sku_id, date)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            logger.info("已重建sku_history表，使用sku_id作为主键字段")
            
            # 创建模型训练日志表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_training_logs (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    sku VARCHAR(50) NOT NULL,
                    training_date DATETIME NOT NULL,
                    metrics JSON,
                    model_version VARCHAR(50),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE KEY unique_training (sku, training_date)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            
            self.connection.commit()
            cursor.close()
            logger.info("数据库表结构检查完成")
            
            return True
            
        except mysql.connector.Error as err:
            logger.error(f"连接MySQL失败 (错误 {err.errno}): {err.msg}")
            return False
        except Exception as e:
            logger.error(f"初始化数据库表结构失败: {str(e)}")
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
            (sku, date, price, confidence)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            date = VALUES(date),
            price = VALUES(price),
            confidence = VALUES(confidence),
            last_updated = CURRENT_TIMESTAMP
            """
            
            # 分批处理大数据集
            records = [
                (row['sku'], row['date'], row['price'], row['confidence'])
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
                (row['sku_id'], row['date'], row['price'], row.get('is_promotion', False))
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

    # 测试数据 - 预测结果
    test_pred = pd.DataFrame({
        'sku': ['test_001', 'test_002'],
        'date': [datetime.now(), datetime.now()],
        'price': [99.9, 199.9],
        'confidence': [0.85, 0.72]
    })

    # 测试数据 - 历史数据
    test_hist = pd.DataFrame({
        'sku_id': ['test_001', 'test_002'],
        'date': ['2023-01-01', '2023-01-02'],
        'price': [89.9, 179.9],
        'is_promotion': [True, False]
    })

    # 执行测试并捕获详细错误
    try:
        logger.info("开始写入测试数据...")
        if writer.write_predictions(test_pred):
            logger.info("预测数据写入成功")
        if writer.write_history(test_hist):
            logger.info("历史数据写入成功")
    except Exception as e:
        logger.error(f"测试失败: {str(e)}", exc_info=True)