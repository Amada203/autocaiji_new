import pandas as pd
import numpy as np
import logging
from impala.dbapi import connect as impala_connect
import mysql.connector
from mysql.connector import Error as MySQLError
from mysql.connector.pooling import MySQLConnectionPool
from functools import wraps
import time

class ImpalaConnector:
    def __init__(self, host='impala-server', port=21050, 
                 database='default', logger=None):
        """初始化Impala连接器"""
        self.host = host
        self.port = port
        self.database = database
        self.logger = logger or logging.getLogger(__name__)
    
    def _connect(self):
        """建立Impala连接"""
        try:
            conn = impala_connect(
                host=self.host,
                port=self.port,
                database=self.database
            )
            return conn
        except Exception as e:
            self.logger.error(f"Impala连接失败: {str(e)}", exc_info=True)
            raise
    
    def _retry(max_retries=3, delay=1):
        """重试装饰器"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                retries = 0
                while retries < max_retries:
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        retries += 1
                        if retries == max_retries:
                            raise
                        time.sleep(delay)
            return wrapper
        return decorator
    
    @_retry()
    def get_historical_data(self, start_date, end_date):
        """获取历史数据"""
        query = """
        SELECT 
            sku_id, 
            date, 
            discount_price
        FROM 
            product_prices
        WHERE 
            date BETWEEN %s AND %s
        ORDER BY 
            sku_id, date
        """
        
        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (start_date, end_date))
                data = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                return pd.DataFrame(data, columns=columns)
        except Exception as e:
            self.logger.error(f"获取历史数据失败: {str(e)}", exc_info=True)
            raise
    
    @_retry()
    def get_latest_data(self, days=7):
        """获取最新数据"""
        query = """
        SELECT 
            sku_id, 
            date, 
            discount_price
        FROM 
            product_prices
        WHERE 
            date >= DATE_SUB(CURRENT_DATE(), INTERVAL %s DAY)
        ORDER BY 
            sku_id, date
        """
        
        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (days,))
                data = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                return pd.DataFrame(data, columns=columns)
        except Exception as e:
            self.logger.error(f"获取最新数据失败: {str(e)}", exc_info=True)
            raise


class MySQLConnector:
    def __init__(self, pool_name='mysql_pool', pool_size=3,
                 host='mysql-server', database='price_prediction',
                 user='user', password='password', logger=None):
        """初始化MySQL连接池"""
        self.pool = None
        self.pool_name = pool_name
        self.pool_size = pool_size
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.logger = logger or logging.getLogger(__name__)
        self._init_pool()
    
    def _init_pool(self):
        """初始化连接池"""
        try:
            self.pool = MySQLConnectionPool(
                pool_name=self.pool_name,
                pool_size=self.pool_size,
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password
            )
        except MySQLError as e:
            self.logger.error(f"MySQL连接池初始化失败: {str(e)}", exc_info=True)
            raise
    
    def _get_connection(self):
        """从连接池获取连接"""
        try:
            return self.pool.get_connection()
        except MySQLError as e:
            self.logger.error(f"获取MySQL连接失败: {str(e)}", exc_info=True)
            raise
    
    def save_model(self, model, model_name):
        """存储模型到MySQL"""
        # 实际实现需要根据模型序列化方式调整
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # 这里简化实现，实际需要序列化模型
                cursor.execute("""
                    INSERT INTO models (name, model_data, created_at)
                    VALUES (%s, %s, NOW())
                    ON DUPLICATE KEY UPDATE
                    model_data = VALUES(model_data),
                    updated_at = NOW()
                """, (model_name, str(model)))
                conn.commit()
        except MySQLError as e:
            self.logger.error(f"存储模型失败: {str(e)}", exc_info=True)
            raise
    
    def load_model(self, model_name):
        """从MySQL加载模型"""
        # 实际实现需要根据模型反序列化方式调整
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor(dictionary=True)
                cursor.execute("""
                    SELECT model_data FROM models 
                    WHERE name = %s
                    ORDER BY updated_at DESC
                    LIMIT 1
                """, (model_name,))
                result = cursor.fetchone()
                if result:
                    # 这里简化实现，实际需要反序列化模型
                    return result['model_data']
                raise ValueError(f"模型 {model_name} 不存在")
        except MySQLError as e:
            self.logger.error(f"加载模型失败: {str(e)}", exc_info=True)
            raise
    
    def save_predictions(self, predictions, table_name):
        """存储预测结果到MySQL"""
        try:
            if not isinstance(predictions, pd.DataFrame):
                raise ValueError("predictions必须是DataFrame")
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # 创建表（如果不存在）
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        sku_id VARCHAR(50) NOT NULL,
                        date DATE NOT NULL,
                        predicted_price DECIMAL(10,2) NOT NULL,
                        confidence FLOAT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE KEY (sku_id, date)
                    )
                """)
                
                # 批量插入数据
                data = [tuple(x) for x in predictions[['sku_id', 'date', 'predicted_price', 'confidence']].values]
                cursor.executemany(f"""
                    INSERT INTO {table_name} (sku_id, date, predicted_price, confidence)
                    VALUES (%s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                    predicted_price = VALUES(predicted_price),
                    confidence = VALUES(confidence),
                    created_at = CURRENT_TIMESTAMP
                """, data)
                conn.commit()
        except Exception as e:
            self.logger.error(f"存储预测结果失败: {str(e)}", exc_info=True)
            raise