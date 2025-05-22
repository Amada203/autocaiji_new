import pandas as pd
import numpy as np
from datetime import datetime
import logging
from dateutil.parser import parse
import argparse
import sys

class DataProcessor:
    def __init__(self, df, logger=None):
        """初始化数据处理器"""
        self._validate_input(df)
        self.df = df.copy()
        self.logger = logger or logging.getLogger(__name__)
        # 保留原始日期用于调试
        self.df['original_date'] = self.df['date'].copy()
    
    def _validate_input(self, df):
        """验证输入数据符合规范"""
        # 支持date或dt作为日期列
        date_col = 'dt' if 'dt' in df.columns else 'date'
        required_cols = {'sku_id', 'discount_price', date_col}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"输入数据必须包含列: {required_cols}")
        if df.empty:
            raise ValueError("输入数据不能为空")
        # 标准化列名
        if 'dt' in df.columns:
            df = df.rename(columns={'dt': 'date'})
        return df
    
    def preprocess_data(self, df):
        """统一预处理流程"""
        # 1. 确保日期格式正确
        df['date'] = pd.to_datetime(df['date'])
        
        # 2. 按SKU和日期排序
        df = df.sort_values(['sku_id', 'date'])
        
        # 3. 补全所有SKU全日期数据并严格按前向后向填充价格
        date_range = pd.date_range(df['date'].min(), df['date'].max())
        
        def fill_missing_dates(group):
            # 创建完整日期范围
            full_dates = pd.DataFrame({'date': date_range})
            # 合并原始数据
            merged = full_dates.merge(group, on='date', how='left')
            # 填充SKU ID
            merged['sku_id'] = group['sku_id'].iloc[0]
            
            # 严格前向优先、后向补充填充价格
            merged['discount_price'] = merged['discount_price'].ffill().bfill()
            
            # 填充促销标志（默认无促销）
            merged['is_promotion'] = merged['is_promotion'].fillna(0)
            
            # 记录填充情况
            filled_count = merged['discount_price'].isna().sum()
            if filled_count > 0:
                self.logger.warning(f"SKU {group['sku_id'].iloc[0]} 仍有 {filled_count} 条记录无法填充价格")
            
            return merged
        
        # 对每个SKU应用填充
        df = df.groupby('sku_id', group_keys=False).apply(fill_missing_dates)
        
        # 严格计算价格变动特征
        df['prev_price'] = df.groupby('sku_id')['discount_price'].shift(1)
        # 严格遵循 t日价格 != t-1日价格 的判断标准
        df['price_change'] = (df['discount_price'] != df['prev_price']).astype(int)
        # 默认没有前一日价格则视为无变动
        df['price_change'] = df['price_change'].fillna(0)
        
        # 验证价格变动计算
        sample_changes = df[df['price_change'] == 1].sample(min(3, len(df)), random_state=42)
        for _, row in sample_changes.iterrows():
            self.logger.debug(
                f"价格变动验证 - SKU {row['sku_id']} {row['date'].date()}: "
                f"{row['prev_price']} → {row['discount_price']}"
            )
        
        self.logger.info(f"补全日期后数据量: {len(df)}条记录 (含填充记录)")
        self.logger.info(
            f"价格变动统计 - 变动次数: {df['price_change'].sum()}, "
            f"占比: {df['price_change'].mean():.2%}, "
            f"平均变动幅度: {df[df['price_change']==1]['discount_price'].mean()-df[df['price_change']==1]['prev_price'].mean():.2f}"
        )
        
        # 4. 计算价格变动特征
        df['prev_price'] = df.groupby('sku_id')['discount_price'].shift(1)
        df['price_change'] = (df['discount_price'] != df['prev_price']).astype(int)
        df['price_change_amount'] = df['discount_price'] - df['prev_price']
        df['price_change_ratio'] = df['price_change_amount'] / df['prev_price'].replace(0, np.nan)
        df['price_change_direction'] = np.sign(df['price_change_amount'])
        
        # 处理边界情况
        df = df.fillna({
            'prev_price': df['discount_price'],
            'price_change': 0,
            'price_change_amount': 0,
            'price_change_ratio': 0,
            'price_change_direction': 0
        })
        
        return df

    def process(self):
        """主处理方法 - 实现完整预处理规范"""
        try:
            # 输出预处理前数据
            self.logger.info("预处理前数据预览:")
            self.logger.info("\n" + str(self.df[['sku_id', 'date', 'discount_price']].head(3)))
            
            # 标准化列名
            if 'dt' in self.df.columns:
                self.df = self.df.rename(columns={'dt': 'date'})
            
            # 执行统一预处理
            invalid_dates = self._clean_data()
            self.df = self.preprocess_data(self.df)
            if len(invalid_dates) > 0:
                self.logger.warning(f"发现 {len(invalid_dates)} 条记录包含无法解析的日期，已过滤")
                self.logger.debug(f"无效日期记录样例:\n{invalid_dates.head()}")
            
            self._calculate_price_features()
            return self.df
        except Exception as e:
            self.logger.error(f"数据处理失败: {str(e)}", exc_info=True)
            raise
    
    def _clean_data(self):
        """完整数据清洗流程"""
        # 重置索引确保唯一性
        self.df = self.df.reset_index(drop=True)

        # 检查索引唯一性
        if not self.df.index.is_unique:
            self.logger.warning("发现重复索引，已重置索引")
            self.df = self.df.reset_index(drop=True)

        # 1. 标准化日期格式 (严格处理输入)
        self.logger.info(f"原始date列样例: {list(self.df['date'].head(3).squeeze())}")
        self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
        invalid_mask = self.df['date'].isna()
        invalid_dates = self.df[invalid_mask].copy().reset_index(drop=True)

        na_count = invalid_mask.sum()
        if na_count > 0:
            sample_errors = list(self.df.loc[invalid_mask, 'date'].head(3).squeeze())
            self.logger.warning(
                f"日期解析失败 {na_count} 条记录，示例: {sample_errors}"
            )

        # 过滤掉无效日期记录
        self.df = self.df[~invalid_mask].copy().reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(f"所有记录的日期都无法解析，请检查数据源。原始date样例: {list(self.df['date'].head(3).squeeze())}")

        # 2. 按SKU和日期排序
        self.df = self.df.sort_values(['sku_id', 'date'])

        # 3. 处理缺失值 (前向后向填充)
        self.df['discount_price'] = (self.df.groupby('sku_id')['discount_price']
                                   .transform(lambda x: x.ffill().bfill()))

        # 4. 移除异常值
        self.df = self.df[(self.df['discount_price'] > 0) & 
                         (self.df['discount_price'] < 1e6)]

        # 5. 计算基础价格特征
        self.df['prev_price'] = self.df.groupby('sku_id')['discount_price'].shift(1)
        self.df['price_change'] = (self.df['discount_price'] != self.df['prev_price']).astype(int)

        return invalid_dates
    
    def _calculate_price_features(self):
        """计算所有价格相关特征"""
        # 1. 价格变动特征
        self.df['price_change_amount'] = self.df['discount_price'] - self.df['prev_price']
        self.df['price_change_ratio'] = self.df['price_change_amount'] / self.df['prev_price'].replace(0, np.nan)
        self.df['price_change_direction'] = np.sign(self.df['price_change_amount'])
        
        # 2. 价格稳定性特征
        self.df['price_stability_7d'] = (self.df.groupby('sku_id')['price_change']
                                       .transform(lambda x: x.rolling(7, min_periods=1).std()))
        
        # 3. 移除临时列
        self.df = self.df.drop(columns=['prev_price', 'original_date'])

def process_dataframe(df, output_path=None):
    """处理DataFrame数据的入口函数"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("开始处理DataFrame数据")

        # 处理数据
        processor = DataProcessor(df)
        processed_df = processor.process()

        # 输出结果
        if args.output:
            processed_df.to_csv(args.output, index=False)
            logger.info(f"结果已保存到: {args.output}")
        else:
            print("\n处理后的数据:")
            print(processed_df)
            
        return processed_df

    except Exception as e:
        logger.error(f"处理失败: {str(e)}", exc_info=True)
        sys.exit(1)

def main():
    """命令行入口点
    使用方式:
    1. 从CSV文件处理:
       python data_processor.py -f input.csv [-o output.csv]
    2. 从DataFrame处理(编程方式):
       from data_processor import process_dataframe
       processed_df = process_dataframe(df)
    """
    parser = argparse.ArgumentParser(description='数据处理器命令行工具')
    parser.add_argument('-f', '--file', help='输入数据文件路径(CSV格式)')
    parser.add_argument('-o', '--output', help='输出文件路径(可选)')
    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    try:
        # 读取数据
        if not args.file:
            logger.error("必须通过 -f/--file 参数指定输入数据文件")
            sys.exit(1)
            
        df = pd.read_csv(args.file)
        logger.info(f"成功读取输入文件: {args.file}")

        # 处理并返回结果
        process_dataframe(df, args.output)

    except Exception as e:
        logger.error(f"处理失败: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()