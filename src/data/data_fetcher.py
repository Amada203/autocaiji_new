import pandas as pd
from impala.dbapi import connect
from typing import Optional, List, Dict
import logging
import numpy as np
import os

class DataFetcher:
    def __init__(self, host: str, port: int, database: str, user: str, password: str):
        """
        初始化数据获取器
        
        Args:
            host: Impala主机地址
            port: Impala端口
            database: 数据库名
            user: 用户名
            password: 密码
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.connection = None
        self.logger = logging.getLogger(__name__)
        
    def connect(self) -> None:
        """建立数据库连接"""
        try:
            self.connection = connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                auth_mechanism='PLAIN'
            )
            self.logger.info("成功连接到Impala")
        except Exception as e:
            self.logger.error(f"连接Impala失败: {str(e)}")
            raise
            
    def disconnect(self) -> None:
        """关闭数据库连接"""
        if self.connection:
            self.connection.close()
            self.logger.info("已断开Impala连接")
            
    def execute_query(self, query: str) -> pd.DataFrame:
        """
        执行SQL查询并返回DataFrame
        
        Args:
            query: SQL查询语句
            
        Returns:
            pd.DataFrame: 查询结果
        """
        if not self.connection:
            self.connect()
            
        try:
            cursor = self.connection.cursor()
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description]
            data = cursor.fetchall()
            return pd.DataFrame(data, columns=columns)
        except Exception as e:
            self.logger.error(f"查询执行失败: {str(e)}")
            raise
            
    def get_cleaning_items(self, base_date: str = '2023-01-01', future_date: str = '2025-03-01') -> List[str]:
        """
        获取需要分析的清洁用品商品ID列表
        
        Args:
            base_date: 基准日期，用于获取商品排名 (YYYY-MM-DD)
            future_date: 未来日期，用于验证商品是否仍然存在 (YYYY-MM-DD)
        
        Returns:
            List[str]: 商品ID列表
        """
        self.logger.info(f"开始获取清洁用品商品列表 (基准日期: {base_date}, 未来日期: {future_date})")
        
        query = f"""
        SELECT DISTINCT CAST(item_id AS BIGINT) as item_id FROM (
            SELECT 
                *,
                ROW_NUMBER() OVER(PARTITION BY platform, category_name ORDER BY item_unit_sales DESC) rn 
            FROM 
                ms_libai.monthly_sales_wide_avg 
            WHERE 
                (
                    category_name RLIKE '洗衣液' OR
                    category_name RLIKE '洗衣粉' OR
                    category_name RLIKE '洗衣皂' OR
                    category_name RLIKE '柔顺'
                )
                AND month_dt = '{base_date}'
        ) a 
        LEFT SEMI JOIN (
            SELECT * FROM ms_libai.monthly_sales_wide_avg 
            WHERE 
                (
                    category_name RLIKE '洗衣液' OR
                    category_name RLIKE '洗衣粉' OR
                    category_name RLIKE '洗衣皂' OR
                    category_name RLIKE '柔顺'
                )
                AND month_dt = '{future_date}'
        ) b 
        ON a.item_id = b.item_id 
        WHERE 
            a.rn <= 300 
            AND LOWER(a.platform) = 'jd'
        """
        
        df = self.execute_query(query)
        self.logger.info(f"成功获取到 {len(df)} 个商品ID")
        return df['item_id'].tolist()
            
    def fetch_cleaning_data(self, 
                           start_date: str, 
                           end_date: str, 
                           base_date: str = '2023-01-01',
                           max_items_per_category: int = 300) -> pd.DataFrame:
        """
        获取清洁用品的价格数据
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            base_date: 基准日期，用于获取商品排名 (YYYY-MM-DD)
            max_items_per_category: 每个类别最多商品数
            
        Returns:
            pd.DataFrame: 包含价格数据的DataFrame
        """
        self.logger.info(f"开始获取清洁用品价格数据 (开始日期: {start_date}, 结束日期: {end_date}, 基准日期: {base_date})")
        
        # 获取每日价格数据
        price_query = f"""
        WITH ranked_items AS (
            SELECT 
                CAST(item_id AS BIGINT) as sku_id,
                category_name,
                ROW_NUMBER() OVER(PARTITION BY category_name ORDER BY item_unit_sales DESC) as category_rank
            FROM 
                ms_libai.monthly_sales_wide_avg
            WHERE 
                month_dt = '{base_date}'
                AND (
                    category_name RLIKE '洗衣液' OR
                    category_name RLIKE '洗衣粉' OR
                    category_name RLIKE '洗衣皂' OR
                    category_name RLIKE '柔顺'
                )
        )
        SELECT 
            CAST(p.sku_id AS BIGINT) as sku_id,
            p.dt as record_date,
            p.page_price as price,
            p.discount_price,
            r.category_name
        FROM 
            ms_libai.jd_daily_price p
        INNER JOIN ranked_items r ON p.sku_id = r.sku_id
        WHERE 
            p.dt BETWEEN '{start_date}' AND '{end_date}'
            AND r.category_rank <= {max_items_per_category}
        """
        
        price_data = self.execute_query(price_query)
        
        # 数据类型转换
        price_data['record_date'] = pd.to_datetime(price_data['record_date'])
        price_data['price'] = price_data['price'].astype(float)
        price_data['discount_price'] = price_data['discount_price'].astype(float)
        
        # 添加日志记录商品数量
        self.logger.info("\n=== 数据统计信息 ===")
        total_records = len(price_data)
        unique_skus = price_data['sku_id'].nunique()
        self.logger.info(f"价格数据表中匹配到 {total_records} 条记录")
        self.logger.info(f"SKU数量: {unique_skus}")
        
        # 按类别统计
        category_stats = price_data.groupby('category_name')['sku_id'].nunique()
        self.logger.info("\n=== 类别统计 ===")
        self.logger.info(f"\n{category_stats}")
        
        return price_data
    
    def fetch_sampling_data(self, 
                           categories: Optional[List[str]] = None,
                           start_date: str = '2022-01-01', 
                           end_date: str = '2024-01-01',
                           max_skus: int = 1000,
                           min_records: int = 30) -> pd.DataFrame:
        """
        获取用于采样策略研究的SKU价格数据，包含各种变化频率的SKU
        
        Args:
            categories: 类别列表，None表示所有类别
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            max_skus: 最大SKU数量
            min_records: 每个SKU最少记录数
            
        Returns:
            pd.DataFrame: 包含价格数据的DataFrame
        """
        self.logger.info(f"开始获取采样策略研究数据 (开始日期: {start_date}, 结束日期: {end_date})")
        
        # 构建类别筛选条件
        category_filter = ""
        if categories:
            category_filter = f"AND m.category_name IN ({', '.join([f'\'{c}\'' for c in categories])})"
        
        # 1. 计算所有SKU的价格变化频率
        change_query = f"""
        WITH price_changes AS (
            SELECT 
                sku_id,
                COUNT(*) as total_records,
                SUM(CASE WHEN page_price != LAG(page_price) OVER(PARTITION BY sku_id ORDER BY dt) THEN 1 ELSE 0 END) as changes,
                SUM(CASE WHEN page_price != LAG(page_price) OVER(PARTITION BY sku_id ORDER BY dt) THEN 1 ELSE 0 END) / COUNT(*) as change_rate
            FROM 
                ms_libai.jd_daily_price
            WHERE 
                dt BETWEEN '{start_date}' AND '{end_date}'
            GROUP BY sku_id
            HAVING total_records >= {min_records}
        )
        SELECT 
            pc.sku_id,
            pc.total_records,
            pc.changes,
            pc.change_rate,
            m.category_name
        FROM 
            price_changes pc
        JOIN ms_libai.monthly_sales_wide_avg m 
            ON pc.sku_id = m.item_id
        WHERE 
            m.month_dt = '{start_date}'
            {category_filter}
        """
        
        changes_df = self.execute_query(change_query)
        
        if len(changes_df) == 0:
            self.logger.warning("未找到符合条件的SKU")
            return pd.DataFrame()
            
        # 2. 按价格变化率分层抽样
        # 确保选取各种变化频率的SKU
        changes_df['change_group'] = pd.qcut(
            changes_df['change_rate'], 
            q=3, 
            labels=['低频变化', '中频变化', '高频变化']
        )
        
        # 从每组中抽取SKU
        selected_skus = []
        group_size = max_skus // 3  # 每组大约取1/3的SKU
        
        for group in ['低频变化', '中频变化', '高频变化']:
            group_df = changes_df[changes_df['change_group'] == group]
            if len(group_df) > 0:
                # 如果该组SKU数量足够，随机抽样
                if len(group_df) > group_size:
                    group_sample = group_df.sample(n=group_size, random_state=42)
                else:
                    group_sample = group_df
                selected_skus.append(group_sample)
        
        if not selected_skus:
            self.logger.warning("分组抽样后没有选取到SKU")
            return pd.DataFrame()
            
        selected_df = pd.concat(selected_skus)
        sku_list = selected_df['sku_id'].tolist()
        
        self.logger.info(f"选取了{len(sku_list)}个SKU用于采样策略研究")
        for group in ['低频变化', '中频变化', '高频变化']:
            count = len(selected_df[selected_df['change_group'] == group])
            self.logger.info(f"  {group}SKU: {count}个")
        
        # 3. 获取这些SKU的价格数据
        sku_values = ", ".join([str(sku) for sku in sku_list])
        price_query = f"""
        SELECT 
            sku_id,
            dt as ds,
            page_price as y,
            discount_price
        FROM 
            ms_libai.jd_daily_price
        WHERE 
            dt BETWEEN '{start_date}' AND '{end_date}'
            AND sku_id IN ({sku_values})
        ORDER BY sku_id, dt
        """
        
        price_data = self.execute_query(price_query)
        
        if len(price_data) == 0:
            self.logger.warning("未获取到价格数据")
            return pd.DataFrame()
            
        # 数据类型转换
        price_data['ds'] = pd.to_datetime(price_data['ds'])
        price_data['y'] = price_data['y'].astype(float)
        
        # 4. 计算价格变化标志（使用严格不等判定）
        price_data['prev_y'] = price_data.groupby('sku_id')['y'].shift(1)
        price_data['change_flag'] = (price_data['y'] != price_data['prev_y']).astype(int)
        
        # 5. 添加分类信息
        price_data = pd.merge(
            price_data,
            selected_df[['sku_id', 'category_name', 'change_group']],
            on='sku_id',
            how='left'
        )
        
        # 记录统计信息
        self.logger.info(f"获取到{len(price_data)}条价格记录")
        self.logger.info(f"平均每个SKU {len(price_data)/len(sku_list):.1f} 条记录")
        self.logger.info(f"价格变化次数: {price_data['change_flag'].sum()}")
        
        return price_data
    
    def generate_sampling_dataset(self, 
                                 categories: Optional[List[str]] = None,
                                 start_date: str = '2022-01-01',
                                 end_date: str = '2024-01-01',
                                 output_path: str = "data/processed/time_series_data.csv") -> pd.DataFrame:
        """
        生成用于采样策略研究的完整数据集
        
        Args:
            categories: 类别列表，None表示所有类别
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            output_path: 输出文件路径
            
        Returns:
            pd.DataFrame: 处理后的数据集
        """
        self.logger.info("开始生成采样策略研究数据集")
        
        # 1. 获取原始数据
        raw_data = self.fetch_sampling_data(
            categories=categories,
            start_date=start_date,
            end_date=end_date,
            max_skus=1000,
            min_records=30
        )
        
        if len(raw_data) == 0:
            raise ValueError("未能获取到足够的测试数据")
        
        # 2. 确保数据完整性 - 对每个SKU创建完整的日期序列
        self.logger.info("处理数据并确保时间序列完整性")
        
        all_dates = pd.date_range(
            start=raw_data['ds'].min(),
            end=raw_data['ds'].max(),
            freq='D'
        )
        
        # 创建完整日期范围的数据框架
        complete_data = []
        
        for sku in raw_data['sku_id'].unique():
            sku_data = raw_data[raw_data['sku_id'] == sku]
            category = sku_data['category_name'].iloc[0]
            change_group = sku_data['change_group'].iloc[0]
            
            # 获取该SKU的所有日期
            sku_dates = pd.DataFrame({'ds': all_dates})
            sku_dates['sku_id'] = sku
            sku_dates['category_name'] = category
            sku_dates['change_group'] = change_group
            
            # 合并现有数据
            merged = pd.merge(
                sku_dates,
                sku_data[['ds', 'sku_id', 'y', 'change_flag']],
                on=['ds', 'sku_id'],
                how='left'
            )
            
            # 填充缺失值（使用前一个有效价格）
            merged['y'] = merged['y'].fillna(method='ffill').fillna(method='bfill')
            merged['change_flag'] = merged['change_flag'].fillna(0)
            
            complete_data.append(merged)
        
        # 合并所有SKU数据
        final_data = pd.concat(complete_data)
        
        # 3. 重新计算价格变化标志（处理填充后的情况）
        grouped = final_data.groupby('sku_id')
        change_flags = []
        
        for _, group in grouped:
            group = group.sort_values('ds')
            group['prev_y'] = group['y'].shift(1)
            group['change_flag'] = (group['y'] != group['prev_y']).astype(int)
            change_flags.append(group)
        
        final_data = pd.concat(change_flags)
        
        # 4. 保存到CSV
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_data.to_csv(output_path, index=False)
        
        # 记录数据统计
        self.logger.info(f"采样策略研究数据集已生成: {output_path}")
        self.logger.info(f"数据统计: {len(final_data)}条记录, {final_data['sku_id'].nunique()}个SKU")
        self.logger.info(f"时间范围: {final_data['ds'].min()} - {final_data['ds'].max()}")
        self.logger.info(f"价格变化次数: {final_data['change_flag'].sum()}")
        
        # 各变化组的统计
        for group in final_data['change_group'].unique():
            group_data = final_data[final_data['change_group'] == group]
            changes = group_data['change_flag'].sum()
            skus = group_data['sku_id'].nunique()
            total = len(group_data)
            self.logger.info(f"  {group}: {skus}个SKU, {changes}次变化, 变化率{changes/total:.4f}")
        
        return final_data
    
    def close(self):
        """
        关闭数据库连接
        """
        self.disconnect() 