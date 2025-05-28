import mysql.connector
import logging
from typing import List, Dict, Any
import pandas as pd
from src.api.main import predict_api, history_api
from src.api.schemas import PredictRequest, PredictBatchRequest

logger = logging.getLogger(__name__)

class PredictionService:
    def __init__(self):
        self.config = {
            'host': 'localhost',
            'port': 13306,
            'user': 'root',
            'password': 'mypassword123',
            'database': 'price_prediction'
        }
        logger.info("初始化PredictionService，数据库配置: %s", {k: v for k, v in self.config.items() if k != 'password'})
        self._test_connection()

    def _test_connection(self):
        """测试数据库连接"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchall()  # 消费所有结果
            cursor.close()
            conn.close()
            logger.info("数据库连接测试成功")
        except Exception as e:
            logger.error("数据库连接测试失败: %s", str(e), exc_info=True)
            raise RuntimeError(f"数据库连接失败: {str(e)}") from e
        
    def _get_connection(self):
        """获取数据库连接"""
        try:
            return mysql.connector.connect(
                **self.config,
                auth_plugin='mysql_native_password'
            )
        except Exception as e:
            logger.error(f"数据库连接失败: {str(e)}")
            raise

    def get_predictions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取预测数据（新增方法）"""
        try:
            logger.info(f"正在查询预测数据，limit={limit}")
            conn = self._get_connection()
            cursor = conn.cursor(dictionary=True)
            
            # 只查询真实的预测数据
            query = """
            SELECT * FROM sku_predictions 
            ORDER BY date DESC 
            LIMIT %s
            """
            cursor.execute(query, (limit,))
            
            results = cursor.fetchall()
            logger.info(f"查询到{len(results)}条预测数据")
            
            cursor.close()
            conn.close()
            
            if not results:
                logger.warning("数据库中没有找到预测数据")
                return []
            
            return results
            
        except Exception as e:
            logger.error(f"查询失败: {str(e)}", exc_info=True)
            raise RuntimeError(f"获取预测数据失败: {str(e)}")

    def get_training_logs(self) -> Dict[str, List[Dict]]:
        """获取所有SKU的训练记录"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor(dictionary=True)
            
            query = """
            SELECT sku, training_date, metrics 
            FROM model_training_logs
            ORDER BY sku, training_date DESC
            """
            cursor.execute(query)
            
            logs = {}
            for row in cursor.fetchall():
                if row['sku'] not in logs:
                    logs[row['sku']] = []
                logs[row['sku']].append({
                    'date': row['training_date'],
                    'metrics': row['metrics']
                })
                
            cursor.close()
            conn.close()
            return logs
            
        except Exception as e:
            logger.error(f"获取训练记录失败: {str(e)}")
            return {}

    def generate_sampling_plan(self, sku: str) -> Dict[str, Any]:
        """生成SKU采样计划"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor(dictionary=True)
            
            # 获取SKU基本信息
            cursor.execute(
                "SELECT * FROM sku_info WHERE sku = %s", 
                (sku,)
            )
            sku_info = cursor.fetchone()
            
            # 获取最近预测数据（使用discount_price）
            cursor.execute(
                """SELECT 
                    sku_id as sku,
                    date as prediction_date,
                    discount_price,
                    probability,
                    predicted_change
                FROM sku_predictions 
                WHERE sku_id = %s 
                ORDER BY date DESC LIMIT 1""",
                (sku,)
            )
            prediction = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            if not prediction:
                logger.warning(f"未找到SKU {sku}的预测数据")
                return {
                    "sku": sku,
                    "status": "no_prediction_data"
                }
            
            return {
                "sku": sku,
                "category": sku_info.get('category'),
                "current_price": prediction.get('discount_price'),
                "change_probability": prediction.get('probability'),
                "predicted_change": prediction.get('predicted_change'),
                "sample_size": self._calculate_sample_size(sku_info),
                "sampling_interval": "weekly",
                "next_sample_date": self._calculate_next_date()
            }
            
        except Exception as e:
            logger.error(f"生成采样计划失败: {str(e)}")
            return {}

    def _calculate_sample_size(self, sku_info: Dict) -> int:
        """计算采样数量(简化示例)"""
        base = 100
        if sku_info['sales_volume'] > 1000:
            return base * 2
        return base

    def _calculate_next_date(self) -> str:
        """计算下次采样日期(简化示例)"""
        from datetime import datetime, timedelta
        return (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")

    def predict_realtime(self, sku_list, days=1, end_date=None):
        """
        实时预测接口，调用新src/api/main.py的predict_batch_api
        """
        reqs = [PredictRequest(sku=sku, date=end_date or pd.Timestamp.now().strftime('%Y-%m-%d')) for sku in sku_list]
        batch_req = PredictBatchRequest(items=reqs)
        results = predict_batch_api(batch_req)
        return [
            {
                "sku_id": r.sku,
                "dates": [r.date],
                "predicted": [r.predict_proba],
                "actual": [None],
            }
            for r in results
        ]

    def _fetch_actuals(self, sku_list, date_list):
        """从数据库获取实际价格，返回dict: (sku, date) -> actual_value"""
        actuals = {}
        try:
            conn = self._get_connection()
            cursor = conn.cursor(dictionary=True)
            for sku in sku_list:
                query = """
                SELECT sku_id, date, discount_price FROM sku_history
                WHERE sku_id = %s AND date IN (%s)
                """
                # 处理IN参数
                date_strs = ','.join(['%s'] * len(date_list))
                full_query = query.replace('(%s)', f'({date_strs})')
                params = [sku] + list(date_list)
                cursor.execute(full_query, params)
                for row in cursor.fetchall():
                    key = (row['sku_id'], row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date']))
                    actuals[key] = row['discount_price']
            cursor.close()
            conn.close()
        except Exception as e:
            logger.warning(f"获取实际值失败: {str(e)}")
        return actuals