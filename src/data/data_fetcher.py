import pandas as pd
import logging
import os
import json
import time
from impala.dbapi import connect
from impala.util import as_pandas
from typing import Dict, Any

class DataFetcher:
    def __init__(self):
        """初始化数据获取器"""
        self.logger = self._setup_logger()
        self.config = self._load_config()
        self.status_file = "logs/pipeline_status.json"
        
    def _get_last_run_date(self):
        """获取上次成功运行的日期"""
        default_date = "2025-04-19"  # 默认起始日期
        try:
            with open(self.status_file, "r") as f:
                status = json.load(f)
                return status.get("last_success_date", default_date)
        except (FileNotFoundError, json.JSONDecodeError):
            return default_date
            
    def _update_last_run_date(self, date):
        """更新最后运行日期"""
        status = {"last_success_date": date}
        with open(self.status_file, "w") as f:
            json.dump(status, f)
        
    def _load_config(self) -> Dict[str, Any]:
        """加载Impala配置"""
        config_path = os.path.join(os.path.dirname(__file__), '../../config/database.json')
        try:
            self.logger.debug(f"加载配置文件: {config_path}")
            with open(config_path) as f:
                config = json.load(f)
                impala_config = config.get('impala', {})
                
                # 关键配置校验
                required_keys = ['host', 'port', 'database']
                missing = [k for k in required_keys if k not in impala_config]
                if missing:
                    raise ValueError(f"缺少必要配置项: {missing}")
                
                self.logger.debug(f"加载的Impala配置: {impala_config}")
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

    def test_connection(self) -> bool:
        """测试Impala连接是否可用"""
        self.logger.info("测试Impala连接...")
        try:
            conn_params = {
                'host': self.config['host'],
                'port': self.config['port'],
                'database': self.config['database'],
                'user': self.config.get('user'),
                'password': self.config.get('password'),
                'auth_mechanism': 'PLAIN',
                'timeout': 5
            }
            self.logger.debug(f"尝试连接参数: {conn_params}")
            
            conn = connect(**conn_params)
            conn.close()
            self.logger.info("Impala连接测试成功")
            return True
        except Exception as e:
            self.logger.error(f"Impala连接测试失败: {str(e)}")
            return False

    def fetch_training_data(self, max_retries=3, retry_delay=60) -> pd.DataFrame:
        """获取训练数据，带重试机制"""
        for attempt in range(max_retries):
            try:
                self.logger.info(f"尝试获取训练数据 (第{attempt+1}次尝试)")
                
                # 增量数据获取 - 只获取上次运行后的新数据
                last_run_date = self._get_last_run_date()  # 获取上次成功运行的日期
                
                query = f"""
                WITH incremental_skus AS (
                    SELECT DISTINCT sku_id
                    FROM jd_daily_price
                    WHERE dt >= '{last_run_date}'
                )
                SELECT 
                    p.sku_id,
                    p.dt AS ds,
                    p.page_price AS y,
                    CASE WHEN p.raw_promotion_detail IS NOT NULL THEN 1 ELSE 0 END AS is_promotion
                FROM jd_daily_price p
                JOIN incremental_skus t ON p.sku_id = t.sku_id
                WHERE p.dt >= '{last_run_date}'
                ORDER BY p.sku_id, p.dt
                """
                
                # 建立连接
                conn_params = {
                    'host': self.config['host'],
                    'port': self.config['port'],
                    'database': self.config['database'],
                    'user': self.config.get('user'),
                    'password': self.config.get('password'),
                    'auth_mechanism': 'PLAIN',
                    'timeout': 60  # 增加超时时间
                }
                
                conn = connect(**conn_params)
                cursor = conn.cursor()
                cursor.execute(query)
                df = as_pandas(cursor)
                
                if df.empty:
                    raise ValueError("获取的数据为空")
                
                # 计算价格变化标志
                df['change_flag'] = (df.groupby('sku_id')['y'].diff() != 0).astype(int)
                
                # 获取最新日期作为下次运行的起始点
                latest_date = df['ds'].max()
                self._update_last_run_date(latest_date)
                
                self.logger.info(f"成功获取 {len(df)} 条记录，最新日期: {latest_date}")
                return df
                
            except Exception as e:
                self.logger.error(f"获取数据失败 (尝试 {attempt+1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    self.logger.info(f"等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                else:
                    raise RuntimeError(f"经过 {max_retries} 次尝试后仍无法获取数据")
            
            if 'conn' in locals():
                conn.close()

    def generate_sampling_dataset(self, categories=None, start_date=None, end_date=None, output_path=None) -> pd.DataFrame:
        """生成采样数据集
        
        参数:
            categories: 可选，商品类别列表
            start_date: 可选，开始日期(YYYY-MM-DD格式)
            end_date: 可选，结束日期(YYYY-MM-DD格式)
            output_path: 可选，保存数据集的路径
            
        返回:
            pandas.DataFrame: 采样数据集
        """
        self.logger.info(f"开始生成采样数据集: start_date={start_date}, end_date={end_date}")
        
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
            
            # 构建基础查询
            query = """
            SELECT 
                sku_id,
                dt AS date,
                page_price,
                discount_price,
                CASE WHEN raw_promotion_detail IS NOT NULL THEN 1 ELSE 0 END AS is_promotion
            FROM jd_daily_price
            WHERE 1=1
            """
            
            # 添加日期条件
            if start_date and end_date:
                query += f" AND dt BETWEEN '{start_date}' AND '{end_date}'"
            elif start_date:
                query += f" AND dt >= '{start_date}'"
            elif end_date:
                query += f" AND dt <= '{end_date}'"
                
            # 添加类别条件
            if categories:
                category_condition = " OR ".join([f"category LIKE '%{cat}%'" for cat in categories])
                query += f" AND ({category_condition})"
            
            cursor = conn.cursor()
            cursor.execute(query)
            df = as_pandas(cursor)
            
            # 保存结果
            if output_path:
                df.to_csv(output_path, index=False)
                self.logger.info(f"数据集已保存至: {output_path}")
                
            self.logger.info(f"成功生成采样数据集，记录数: {len(df)}")
            return df
            
        except Exception as e:
            self.logger.error(f"生成采样数据集失败: {str(e)}")
            raise
        finally:
            if 'conn' in locals():
                conn.close()

    def get_recent_data(self, days=30) -> pd.DataFrame:
        """获取最近N天的价格数据用于验证
        
        Args:
            days: 要获取的天数
            
        Returns:
            pd.DataFrame: 包含日期(ds)、价格(y)、商品ID(sku_id)和变化标志(change_flag)的数据
        """
        self.logger.info(f"开始获取最近{days}天的价格数据")
        
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
            
            # 执行查询获取最近N天的原始价格数据
            query = f"""
            WITH recent_data AS (
                SELECT 
                    sku_id,
                    dt AS ds,
                    page_price AS y,
                    LAG(page_price) OVER (PARTITION BY sku_id ORDER BY dt) AS prev_price
                FROM jd_daily_price
                WHERE dt >= DATE_SUB(FROM_UNIXTIME(UNIX_TIMESTAMP()), {days})
            )
            SELECT 
                sku_id,
                ds,
                y,
                CASE 
                    WHEN prev_price IS NULL THEN 0
                    WHEN y != prev_price THEN 1
                    ELSE 0
                END AS change_flag
            FROM recent_data
            ORDER BY sku_id, ds
            """
            
            cursor = conn.cursor()
            cursor.execute(query)
            df = as_pandas(cursor)
            
            self.logger.info(f"成功获取 {len(df)} 条最近{days}天的价格记录")
            return df
            
        except Exception as e:
            self.logger.error(f"获取最近价格数据失败: {str(e)}")
            raise
        finally:
            if 'conn' in locals():
                conn.close()

    def close(self):
        """关闭资源"""
        self.logger.info("DataFetcher资源清理完成")

if __name__ == "__main__":
    # 测试数据获取
    fetcher = DataFetcher()
    try:
        data = fetcher.fetch_training_data()
        print(f"获取数据成功，记录数: {len(data)}")
    except Exception as e:
        print(f"测试失败: {str(e)}")