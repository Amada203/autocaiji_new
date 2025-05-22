import pandas as pd
import mysql.connector
from datetime import datetime
from typing import List
import logging
import json
import os
from mysql.connector import Error as MySQLError

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MySQLWriter:
    def __init__(self, host='localhost', user='root', 
                 password='mypassword123', database='price_prediction', port=13306):
        """初始化MySQL写入器
        
        Args:
            host: 数据库主机地址
            user: 数据库用户名
            password: 数据库密码
            database: 数据库名称
            port: 数据库端口
        """
        self.connection = None
        self.db_config = {
            'host': host,
            'user': user,
            'password': password,
            'database': database,
            'port': port
        }
        
    def connect(self):
        """连接到MySQL数据库"""
        try:
            self.connection = mysql.connector.connect(
                **self.db_config,
                connect_timeout=30  # 30秒连接超时
            )
            self.connection.autocommit = False  # 使用事务模式
            logger.info("成功连接到MySQL数据库")
            return True
        except mysql.connector.Error as err:
            logger.error(f"数据库连接失败 (错误 {err.errno}): {err.msg}")
            logger.info("请检查以下配置是否正确:")
            logger.info(f"主机: {self.db_config['host']}")
            logger.info(f"端口: {self.db_config['port']}")
            logger.info(f"数据库: {self.db_config['database']}")
            logger.info(f"用户名: {self.db_config['user']}")
            return False

    def write_predictions(self, predictions_df: pd.DataFrame, rebuild_table=False) -> bool:
        """写入预测数据到sku_predictions表，使用批量插入提高性能
        
        Args:
            predictions_df: 包含预测数据的DataFrame，需包含以下字段:
                - sku_id: SKU标识
                - date: 预测日期
                - discount_price: 折扣价格
                - probability: 价格变动概率 (0-1)
                - predicted_change: 预测是否变动 (0或1)
            rebuild_table: 是否重建表，默认为False（增量更新）
        """
        # 强制限制为1000条
        predictions_df = predictions_df.head(1000)
        
        # 验证必需字段
        required_fields = ['sku_id', 'date', 'discount_price', 'probability', 'predicted_change']
        missing_fields = [field for field in required_fields if field not in predictions_df.columns]
        if missing_fields:
            logger.error(f"缺少必需字段: {missing_fields}")
            return False
            
        if not self.connect():
            return False

        try:
            cursor = self.connection.cursor()
            
            # 设置会话超时
            cursor.execute("SET SESSION wait_timeout=300")  # 5分钟超时
            cursor.execute("SET SESSION innodb_lock_wait_timeout=50")  # 50秒锁等待超时
            
            # 检查表是否存在
            cursor.execute("SHOW TABLES LIKE 'sku_predictions'")
            table_exists = cursor.fetchone() is not None
            
            if rebuild_table or not table_exists:
                if table_exists:
                    # 如果表存在且选择重建，先备份数据
                    logger.info("备份现有预测数据...")
                    try:
                        cursor.execute("CREATE TABLE IF NOT EXISTS sku_predictions_backup LIKE sku_predictions")
                        cursor.execute("TRUNCATE TABLE sku_predictions_backup")
                        cursor.execute("INSERT INTO sku_predictions_backup SELECT * FROM sku_predictions")
                        logger.info("预测数据备份完成")
                    except Exception as e:
                        logger.warning(f"备份数据失败: {str(e)}")
                    
                    # 清空表而不是删除重建
                    logger.info("清空sku_predictions表...")
                    cursor.execute("TRUNCATE TABLE sku_predictions")
                else:
                    # 表不存在，创建新表
                    logger.info("创建sku_predictions表...")
                    cursor.execute("""
                        CREATE TABLE sku_predictions (
                            sku_id VARCHAR(50),
                            date DATE,
                            discount_price DECIMAL(10,2),
                            probability FLOAT,
                            predicted_change TINYINT,
                            PRIMARY KEY (sku_id, date),
                            INDEX idx_date (date),
                            INDEX idx_sku (sku_id)
                        ) ENGINE=InnoDB
                    """)
                logger.info("表准备完成")
            else:
                logger.info("使用现有sku_predictions表（增量更新模式）")

            # 准备批量插入（使用REPLACE INTO支持增量更新）
            insert_sql = """
                REPLACE INTO sku_predictions (sku_id, date, discount_price, probability, predicted_change)
                VALUES (%s, %s, %s, %s, %s)
            """
            
            # 批量处理，每1000条提交一次
            batch_size = 1000
            records = []
            success_count = 0
            total_records = len(predictions_df)
            
            # 处理每行数据
            for _, row in predictions_df.iterrows():
                try:
                    # 处理日期字段
                    date_value = row['date']
                    if hasattr(date_value, 'strftime'):
                        date_str = date_value.strftime('%Y-%m-%d')
                    else:
                        date_str = str(date_value)
                    
                    # 验证数据
                    probability = float(row['probability'])
                    if not 0 <= probability <= 1:
                        logger.warning(f"SKU {row['sku_id']} 的概率值超出范围: {probability}")
                        probability = max(0, min(1, probability))  # 强制限制在0-1之间
                    
                    record = (
                        str(row['sku_id']),
                        date_str,
                        float(row['discount_price']),
                        probability,
                        int(bool(row['predicted_change']))  # 确保是0或1
                    )
                    records.append(record)
                    
                    # 每达到批量大小就执行一次批量插入
                    if len(records) >= batch_size:
                        cursor.executemany(insert_sql, records)
                        success_count += len(records)
                        self.connection.commit()  # 提交事务
                        records = []
                        logger.info(f"已处理 {success_count}/{total_records} 条记录 ({(success_count/total_records*100):.2f}%)")
                        
                except Exception as e:
                    logger.error(f"处理行数据时出错 - SKU: {row.get('sku_id', 'N/A')}, 错误: {str(e)}")
                    continue

            # 处理剩余记录
            if records:
                cursor.executemany(insert_sql, records)
                success_count += len(records)
                self.connection.commit()
            
            logger.info(f"成功写入 {success_count}/{total_records} 条记录")
            return success_count > 0

        except MySQLError as e:
            logger.error(f"MySQL错误: {str(e)}", exc_info=True)
            if self.connection:
                self.connection.rollback()
            return False
        except Exception as e:
            logger.error(f"写入预测结果失败: {str(e)}", exc_info=True)
            if self.connection:
                self.connection.rollback()
            return False
        finally:
            if self.connection and self.connection.is_connected():
                cursor.close()
                self.connection.close()

    def write_history(self, history_df: pd.DataFrame, rebuild_table=False, limit=None) -> bool:
        """写入历史数据到sku_history表，使用批量插入提高性能
        
        Args:
            history_df: 包含历史数据的DataFrame
            rebuild_table: 是否重建表，默认为False（增量更新）
            limit: 限制处理的记录数，默认为None（处理所有记录）
        """
        # 1. 定义字段映射关系
        field_mapping = {
            'ds': 'date',      # 将 'ds' 映射到 'date'
            'y': 'price'       # 将 'y' 映射到 'price'
        }
        
        # 2. 定义需要保留的字段
        required_fields = ['sku_id', 'date', 'price', 'is_promotion']
        
        # 3. 处理数据
        df = history_df.copy()
        # 进行字段映射
        df = df.rename(columns=field_mapping)
        
        # 4. 验证所需字段是否存在
        missing_fields = [field for field in required_fields if field not in df.columns]
        if missing_fields:
            logger.error(f"缺少必需字段: {missing_fields}")
            return False
            
        # 5. 只选择需要的字段，自动排除 change_flag
        df = df[required_fields]
        
        if not self.connect():
            return False

        try:
            cursor = self.connection.cursor()
            
            # 设置会话超时
            cursor.execute("SET SESSION wait_timeout=300")
            cursor.execute("SET SESSION innodb_lock_wait_timeout=50")
            
            # 检查表是否存在
            cursor.execute("SHOW TABLES LIKE 'sku_history'")
            table_exists = cursor.fetchone() is not None
            
            if rebuild_table or not table_exists:
                if table_exists:
                    # 如果表存在且选择重建，先备份数据
                    logger.info("备份现有历史数据...")
                    try:
                        cursor.execute("CREATE TABLE IF NOT EXISTS sku_history_backup LIKE sku_history")
                        cursor.execute("TRUNCATE TABLE sku_history_backup")
                        cursor.execute("INSERT INTO sku_history_backup SELECT * FROM sku_history")
                        logger.info("历史数据备份完成")
                    except Exception as e:
                        logger.warning(f"备份数据失败: {str(e)}")
                    
                    # 清空表而不是删除重建
                    logger.info("清空sku_history表...")
                    cursor.execute("TRUNCATE TABLE sku_history")
                else:
                    # 表不存在，创建新表
                    logger.info("创建sku_history表...")
                    cursor.execute("""
                        CREATE TABLE sku_history (
                            sku_id VARCHAR(50) PRIMARY KEY,
                            date DATE,
                            price DECIMAL(10,2),
                            is_promotion BOOLEAN
                        ) ENGINE=InnoDB
                    """)
                logger.info("表准备完成")
            else:
                logger.info("使用现有sku_history表（增量更新模式）")

            # 准备批量插入（使用REPLACE INTO支持增量更新）
            insert_sql = """
                REPLACE INTO sku_history (sku_id, date, price, is_promotion)
                VALUES (%s, %s, %s, %s)
            """
            
            # 批量处理，每1000条提交一次
            batch_size = 1000
            records = []
            success_count = 0
            total_records = len(history_df)

            # 如果设置了limit，限制处理的记录数
            if limit is not None and limit > 0:
                logger.info(f"限制处理前 {limit} 条记录（总计 {total_records} 条）")
                history_df = history_df.head(limit)
                total_records = len(history_df)

            # 处理每行数据 - 使用处理后的df而不是原始history_df
            for _, row in df.iterrows():
                try:
                    # 处理日期字段
                    date_value = row['date']
                    if hasattr(date_value, 'strftime'):
                        date_str = date_value.strftime('%Y-%m-%d')
                    elif isinstance(date_value, str):
                        # 尝试解析日期字符串
                        try:
                            date_obj = datetime.strptime(date_value, '%Y-%m-%d')
                            date_str = date_obj.strftime('%Y-%m-%d')
                        except ValueError:
                            logger.error(f"无效的日期格式: {date_value}, 行数据: {row.to_dict()}")
                            continue
                    else:
                        logger.error(f"不支持的日期类型: {type(date_value)}, 值: {date_value}")
                        continue
                    
                    # 验证其他字段
                    try:
                        sku_id = str(row['sku_id'])
                        price = float(row['price'])
                        is_promotion = bool(row['is_promotion'])
                    except (KeyError, ValueError, TypeError) as e:
                        logger.error(f"字段验证失败: {str(e)}, 行数据: {row.to_dict()}")
                        continue
                    
                    record = (sku_id, date_str, price, is_promotion)
                    records.append(record)
                    
                    # 每达到批量大小就执行一次批量插入
                    if len(records) >= batch_size:
                        cursor.executemany(insert_sql, records)
                        success_count += len(records)
                        self.connection.commit()
                        records = []
                        logger.info(f"已处理 {success_count}/{total_records} 条记录 ({(success_count/total_records*100):.2f}%)")
                        
                except Exception as e:
                    logger.error(f"处理历史数据时出错: {str(e)}, 行数据: {row.to_dict() if hasattr(row, 'to_dict') else row}")
                    continue

            # 处理剩余记录
            if records:
                cursor.executemany(insert_sql, records)
                success_count += len(records)
                self.connection.commit()

            logger.info(f"已写入 {success_count} 条历史记录 (总计: {success_count}/{total_records})")
            return success_count > 0

        except MySQLError as e:
            logger.error(f"MySQL错误: {str(e)}", exc_info=True)
            if self.connection:
                self.connection.rollback()
            return False
        except Exception as e:
            logger.error(f"写入历史数据失败: {str(e)}", exc_info=True)
            if self.connection:
                self.connection.rollback()
            return False
        finally:
            if self.connection and self.connection.is_connected():
                cursor.close()
                self.connection.close()

    def query_predictions(self, sku_ids: List[str] = None) -> pd.DataFrame:
        """查询预测数据"""
        if not self.connect():
            return pd.DataFrame()
        
        try:
            cursor = self.connection.cursor(dictionary=True)
            
            if sku_ids:
                query = "SELECT * FROM sku_predictions WHERE sku_id IN (%s)"
                in_clause = ','.join(['%s'] * len(sku_ids))
                cursor.execute(query % in_clause, sku_ids)
            else:
                cursor.execute("SELECT * FROM sku_predictions LIMIT 100")
                
            results = cursor.fetchall()
            return pd.DataFrame(results)
            
        except Exception as e:
            logger.error(f"查询预测数据失败: {str(e)}")
            return pd.DataFrame()
        finally:
            if self.connection and self.connection.is_connected():
                cursor.close()
                self.connection.close()

if __name__ == "__main__":
    # 尝试从database.json读取配置
    db_config = {
        'host': 'localhost',
        'port': 13306,  # 使用Docker映射的端口
        'database': 'price_prediction',
        'user': 'root',
        'password': 'mypassword123'
    }

    # 配置加载和验证
    possible_paths = [
        'database.json',  # 当前目录
        '../database.json',  # 上级目录
        'config/database.json',  # config子目录
        os.path.expanduser('~/database.json')  # 用户主目录
    ]
    
    config_path = None
    for path in possible_paths:
        if os.path.exists(path):
            config_path = path
            break
            
    try:
        if not config_path:
            error_msg = "未找到database.json配置文件，请检查以下位置:\n"
            error_msg += "\n".join(f" - {os.path.abspath(p)}" for p in possible_paths)
            error_msg += "\n请将配置文件放在上述任一位置"
            raise FileNotFoundError(error_msg)
        
        with open(config_path) as f:
            config_data = json.load(f)
            
        if 'mysql' not in config_data:
            raise ValueError("配置文件中缺少mysql节点")
            
        mysql_config = config_data['mysql']
        required_keys = ['host', 'port', 'database', 'user', 'password']
        missing_keys = [k for k in required_keys if k not in mysql_config]
        if missing_keys:
            raise ValueError(f"配置缺少必要参数: {missing_keys}")
        
        # 使用配置文件的参数
        db_config = {
            'host': mysql_config['host'],
            'port': int(mysql_config['port']),
            'database': mysql_config['database'],
            'user': mysql_config['user'],
            'password': mysql_config['password']
        }
        
        logger.info("成功加载数据库配置:")
        logger.info(f"  主机: {db_config['host']}")
        logger.info(f"  端口: {db_config['port']}")
        logger.info(f"  数据库: {db_config['database']}")
        logger.info(f"  用户名: {db_config['user']}")
        
    except Exception as e:
        logger.error(f"加载数据库配置失败: {str(e)}")
        logger.info("请检查database.json文件内容和格式")
        logger.info("示例格式:")
        logger.info('''{
    "mysql": {
        "host": "localhost",
        "port": 13306,
        "user": "root",
        "password": "mypassword123",
        "database": "price_prediction"
    }
}''')
        raise

    # 创建写入器实例
    writer = MySQLWriter(**db_config)

    # 测试数据 - 预测结果
    test_pred = pd.DataFrame({
        'sku': ['test_001', 'test_002'],  # 使用'sku'而不是'sku_id'
        'date': [datetime.now().date(), datetime.now().date()],
        'price': [99.9, 199.9],
        'confidence': [0.85, 0.72]
    })
    logger.info("测试预测数据字段: %s", test_pred.columns.tolist())

    # 测试数据 - 历史数据
    test_hist = pd.DataFrame({
        'sku_id': ['test_001', 'test_002'],
        'date': ['2023-01-01', '2023-01-02'],
        'price': [89.9, 179.9],
        'is_promotion': [True, False]
    })

    try:
        logger.info("开始写入测试数据...")
        
        # 写入测试数据
        if writer.write_predictions(test_pred):
            logger.info("预测数据写入成功")
            # 立即查询验证
            df = writer.query_predictions(['test_001', 'test_002'])
            logger.info(f"查询结果: {len(df)} 条记录")
            if not df.empty:
                logger.info("前5条记录:\n%s", df.head().to_string())
        
        if writer.write_history(test_hist):
            logger.info("历史数据写入成功")
            
    except Exception as e:
        logger.error(f"测试失败: {str(e)}", exc_info=True)