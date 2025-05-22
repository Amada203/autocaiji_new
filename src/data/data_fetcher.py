import pandas as pd
import numpy as np
import logging
import os
import json
import time
import sys
from datetime import datetime, timedelta
from impala.dbapi import connect
from impala.util import as_pandas
from typing import Dict, Any, Optional, Union, List, Set, Tuple
from pathlib import Path

class DataFetcher:
    def __init__(self):
        """初始化数据获取器"""
        self.logger = self._setup_logger()
        self.config = self._load_config()
        self.status_file = "logs/pipeline_status.json"
        self.earliest_date = "2023-01-01"
        self.backup_data_path = "/Users/ruixue.li/lrx/automore/autocaiji/data/query-impala-1632023.csv"

    def _get_last_run_date(self) -> str:
        """获取上次成功运行的日期"""
        default_date = self.earliest_date
        try:
            with open(self.status_file, "r") as f:
                status = json.load(f)
                return status.get("last_success_date", default_date)
        except (FileNotFoundError, json.JSONDecodeError):
            return default_date
            
    def _update_last_run_date(self, date: str) -> None:
        """更新最后运行日期"""
        os.makedirs(os.path.dirname(self.status_file), exist_ok=True)
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
                
                required_keys = ['host', 'port', 'database']
                missing = [k for k in required_keys if k not in impala_config]
                if missing:
                    raise ValueError(f"缺少必要配置项: {missing}")
                
                self.logger.debug(f"加载的Impala配置: {impala_config}")
                return impala_config
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {str(e)}")
            raise

    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('DataFetcher')
        logger.setLevel(logging.INFO)
        
        os.makedirs('logs', exist_ok=True)
        
        file_handler = logging.FileHandler('logs/data_fetcher.log')
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            '%(levelname)s - %(message)s'
        ))
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        return logger

    def _get_connection(self, timeout: int = 30) -> connect:
        """获取Impala连接"""
        conn_params = {
            'host': self.config['host'],
            'port': self.config['port'],
            'database': self.config['database'],
            'user': self.config.get('user'),
            'password': self.config.get('password'),
            'auth_mechanism': 'PLAIN',
            'timeout': timeout
        }
        return connect(**conn_params)

    def test_connection(self) -> bool:
        """测试Impala连接是否可用"""
        self.logger.info("测试Impala连接...")
        try:
            conn = self._get_connection(timeout=5)
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            conn.close()
            self.logger.info("Impala连接测试成功")
            return True
        except Exception as e:
            self.logger.error(f"Impala连接测试失败: {str(e)}")
            return False

    def _get_valid_skus(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Set[int]:
        """获取数据完整性符合要求的SKU列表"""
        if start_date is None:
            start_date = self.earliest_date
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        self.logger.info(f"开始检查SKU数据完整性 ({start_date} 至 {end_date})")
        
        conn = None
        try:
            conn = self._get_connection(timeout=60)
            
            completeness_query = """
            WITH date_range AS (
                SELECT DISTINCT dt FROM jd_daily_price 
                WHERE dt BETWEEN %(start_date)s AND %(end_date)s
            ),
            expected_counts AS (
                SELECT COUNT(*) as total_days FROM date_range
            ),
            sku_completeness AS (
                SELECT 
                    p.sku_id,
                    COUNT(*) as actual_records,
                    e.total_days as expected_records,
                    (COUNT(*) * 100.0 / e.total_days) as completeness_ratio
                FROM jd_daily_price p
                CROSS JOIN expected_counts e
                WHERE dt BETWEEN %(start_date)s AND %(end_date)s
                GROUP BY p.sku_id, e.total_days
                HAVING completeness_ratio >= 90
            )
            SELECT 
                sku_id,
                actual_records,
                expected_records,
                completeness_ratio
            FROM sku_completeness
            ORDER BY completeness_ratio DESC
            """
            
            cursor = conn.cursor()
            cursor.execute(completeness_query, {
                'start_date': start_date,
                'end_date': end_date
            })
            
            completeness_df = as_pandas(cursor)
            cursor.close()
            
            if completeness_df.empty:
                self.logger.warning("未找到符合完整性要求的SKU")
                return set()
            
            avg = completeness_df['completeness_ratio'].mean()
            min_ratio = completeness_df['completeness_ratio'].min()
            max_ratio = completeness_df['completeness_ratio'].max()
            
            self.logger.info(f"SKU完整性检查完成 - 总数: {len(completeness_df)}, 完整率: {avg:.2f}% (范围: {min_ratio:.2f}%-{max_ratio:.2f}%)")
            
            report_path = "data/raw/sku_completeness_report.csv"
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            completeness_df.to_csv(report_path, index=False)
            self.logger.info(f"报告保存至: {report_path}")
            
            return set(completeness_df['sku_id'].astype(int))
            
        except Exception as e:
            self.logger.error(f"获取有效SKU列表失败: {str(e)}")
            raise
        finally:
            if conn is not None:
                conn.close()

    def _load_backup_data(self) -> pd.DataFrame:
        """从备用CSV文件加载数据"""
        self.logger.info(f"从备用文件加载数据: {self.backup_data_path}")
        try:
            if not os.path.exists(self.backup_data_path):
                raise FileNotFoundError(f"备用数据文件不存在: {self.backup_data_path}")
                
            df = pd.read_csv(self.backup_data_path)
            
            required_cols = ['sku_id', 'dt', 'discount_price']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"备用数据文件缺少必要的列: {missing_cols}")
                
            # 统一列名处理，确保只保留date列
            if 'dt' in df.columns:
                df = df.rename(columns={'dt': 'date'})
            if 'date' in df.columns and 'dt' in df.columns:
                df = df.drop(columns=['dt'])
                
            if 'is_promotion' not in df.columns:
                df['is_promotion'] = 0
                
            self.logger.info(f"成功从备用文件加载数据，记录数: {len(df)}")
            return df
            
        except Exception as e:
            self.logger.error(f"加载备用数据失败: {str(e)}")
            raise

    def fetch_training_data(self,
                          train_end: Optional[str] = None,
                          val_end: Optional[str] = None,
                          test_end: Optional[str] = None,
                          max_retries: int = 1,
                          retry_delay: int = 20) -> Dict[str, pd.DataFrame]:
        """获取训练数据，按时间轴划分数据集"""
        if not all([train_end, val_end, test_end]):
            # 计算上个月月末日期作为默认test_end
            today = datetime.now()
            first_of_month = today.replace(day=1)
            last_month_end = first_of_month - timedelta(days=1)
            test_end = last_month_end.strftime('%Y-%m-%d')
            
            # 计算验证集和训练集结束日期
            val_end = (last_month_end - timedelta(days=30)).strftime('%Y-%m-%d')
            train_end = (last_month_end - timedelta(days=90)).strftime('%Y-%m-%d')

        try:
            datetime.strptime(train_end, '%Y-%m-%d')
            datetime.strptime(val_end, '%Y-%m-%d')
            datetime.strptime(test_end, '%Y-%m-%d')
            
            if not (self.earliest_date <= train_end < val_end < test_end):
                raise ValueError(
                    f"日期顺序错误: {self.earliest_date} <= {train_end} < {val_end} < {test_end}"
                )
        except ValueError as e:
            self.logger.error(f"日期格式或顺序错误: {str(e)}")
            raise

        for attempt in range(max_retries):
            conn = None
            try:
                self.logger.info(f"尝试获取训练数据 (第{attempt+1}次尝试)")
                
                valid_skus = self._get_valid_skus(self.earliest_date, test_end)
                if not valid_skus:
                    raise ValueError("没有找到符合完整性要求的SKU")
                
                self.logger.info(f"找到 {len(valid_skus)} 个符合完整性要求的SKU")
                
                queries = {
                    'train': """
                        SELECT 
                            sku_id,
                            dt,
                            discount_price,
                            CASE WHEN raw_promotion_detail IS NOT NULL THEN 1 ELSE 0 END AS is_promotion
                        FROM jd_daily_price
                        WHERE dt >= %(start_date)s AND dt <= %(end_date)s
                        AND sku_id IN %(valid_skus)s
                        ORDER BY sku_id, dt
                    """,
                    'val': """
                        SELECT 
                            sku_id,
                            dt AS `date`,
                            discount_price,
                            CASE WHEN raw_promotion_detail IS NOT NULL THEN 1 ELSE 0 END AS is_promotion
                        FROM jd_daily_price
                        WHERE dt > %(start_date)s AND dt <= %(end_date)s
                        AND sku_id IN %(valid_skus)s
                        ORDER BY sku_id, dt
                    """,
                    'test': """
                        SELECT 
                            sku_id,
                            dt AS `date`,
                            discount_price,
                            CASE WHEN raw_promotion_detail IS NOT NULL THEN 1 ELSE 0 END AS is_promotion
                        FROM jd_daily_price
                        WHERE dt > %(start_date)s AND dt <= %(end_date)s
                        AND sku_id IN %(valid_skus)s
                        ORDER BY sku_id, dt
                    """
                }
                
                date_ranges = {
                    'train': (self.earliest_date, train_end),
                    'val': (train_end, val_end),
                    'test': (val_end, test_end)
                }
                
                conn = self._get_connection(timeout=60)
                datasets = {}
                
                for name, (start, end) in date_ranges.items():
                    cursor = conn.cursor()
                    # 确保只保留date列，移除dt列
                    cursor.execute(queries[name], {
                        'start_date': start,
                        'end_date': end,
                        'valid_skus': tuple(valid_skus)
                    })
                    df = as_pandas(cursor)
                    cursor.close()
                    
                    if not df.empty:
                        # 统一列名处理：优先使用date列，如果没有则使用dt列
                        if 'date' not in df.columns and 'dt' in df.columns:
                            df = df.rename(columns={'dt': 'date'})
                        elif 'date' in df.columns and 'dt' in df.columns:
                            df = df.drop(columns=['dt'])
                    
                    if df.empty:
                        self.logger.warning(f"{name}数据集为空")
                    else:
                        min_date = df['date'].min()
                        max_date = df['date'].max()
                        if name == 'train':
                            if not (self.earliest_date <= min_date <= max_date <= train_end):
                                raise ValueError(f"训练集日期范围错误: {min_date} 到 {max_date}")
                        elif name == 'val':
                            if not (train_end < min_date <= max_date <= val_end):
                                raise ValueError(f"验证集日期范围错误: {min_date} 到 {max_date}")
                        else:
                            if not (val_end < min_date <= max_date <= test_end):
                                raise ValueError(f"测试集日期范围错误: {min_date} 到 {max_date}")
                        
                        self.logger.info(
                            f"{name}数据集 - 记录数: {len(df)}, SKU数: {df['sku_id'].nunique()}, "
                            f"日期范围: {df['date'].min()}至{df['date'].max()}, "
                            f"平均价格: {df['discount_price'].mean():.2f}"
                        )
                        
                        if not df.empty:
                            first_row = df.iloc[0][['sku_id', 'date', 'discount_price', 'is_promotion']].to_dict()
                            self.logger.info(
                                f"{name}数据集首行预览 - " +
                                ", ".join([f"{k}: {v}" for k, v in first_row.items()])
                            )
                    
                    datasets[name] = df
                
                if not datasets['train'].empty and not datasets['val'].empty:
                    train_dates = set(datasets['train']['date'])
                    val_dates = set(datasets['val']['date'])
                    if train_dates & val_dates:
                        raise ValueError(f"训练集和验证集日期有重叠: {train_dates & val_dates}")
                
                if not datasets['val'].empty and not datasets['test'].empty:
                    val_dates = set(datasets['val']['date'])
                    test_dates = set(datasets['test']['date'])
                    if val_dates & test_dates:
                        raise ValueError(f"验证集和测试集日期有重叠: {val_dates & test_dates}")
                
                latest_date = max(df['date'].max() for df in datasets.values() if not df.empty)
                self._update_last_run_date(latest_date)
                
                self.logger.info(f"成功获取所有数据集，最新日期: {latest_date}")
                
                self.logger.info(
                    f"数据集划分 - 训练集: {self.earliest_date}至{train_end}, "
                    f"验证集: {train_end}至{val_end}, "
                    f"测试集: {val_end}至{test_end}"
                )
                
                return datasets
                
            except Exception as e:
                self.logger.error(f"获取数据失败 (尝试 {attempt+1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    self.logger.info(f"等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                else:
                    raise RuntimeError(f"经过 {max_retries} 次尝试后仍无法获取数据")
            finally:
                if conn is not None:
                    conn.close()

    def close(self):
        """关闭资源"""
        self.logger.info("DataFetcher资源清理完成")

def main():
    """主函数，用于测试数据获取功能"""
    fetcher = DataFetcher()
    try:
        if not fetcher.test_connection():
            raise RuntimeError("数据库连接测试失败")
            
        datasets = fetcher.fetch_training_data()
        
        for name, data in datasets.items():
            print(f"\n{name.upper()}数据集统计:")
            print(f"记录数: {len(data)}")
            print(f"SKU数量: {data['sku_id'].nunique()}")
            print(f"日期范围: {data['date'].min()} 至 {data['date'].max()}")
            
    except Exception as e:
        print(f"测试失败: {str(e)}")
        sys.exit(1)
    finally:
        fetcher.close()

if __name__ == "__main__":
    main()