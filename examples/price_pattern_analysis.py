import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PricePatternAnalyzer:
    def __init__(self, df: pd.DataFrame):
        """
        初始化价格模式分析器
        
        Args:
            df: 包含价格数据的DataFrame
        """
        self.df = df.copy()
        self._preprocess_data()
        
    def _preprocess_data(self):
        """数据预处理"""
        # 确保日期格式正确
        self.df['record_date'] = pd.to_datetime(self.df['record_date'])
        
        # 计算价格变动
        self.df = self.df.sort_values(['sku_id', 'record_date'])
        self.df['prev_price'] = self.df.groupby('sku_id')['price'].shift(1)
        self.df['price_change_flag'] = (self.df['price'] != self.df['prev_price']).astype(int)
        self.df['price_change_amount'] = self.df['price'] - self.df['prev_price']
        self.df['price_change_ratio'] = self.df['price_change_amount'] / self.df['prev_price']
        self.df['price_change_direction'] = np.sign(self.df['price_change_amount'])
        
        # 添加时间特征
        self.df['weekday'] = self.df['record_date'].dt.weekday
        self.df['month'] = self.df['record_date'].dt.month
        self.df['day'] = self.df['record_date'].dt.day
        
    def analyze_weekly_pattern(self) -> pd.DataFrame:
        """
        分析每周价格变动模式
        
        Returns:
            pd.DataFrame: 每周价格变动统计
        """
        # 计算每个SKU的每周价格变动频率
        weekly_pattern = self.df.groupby(['sku_id', 'weekday'])['price_change_flag'].mean().reset_index()
        weekly_pattern = weekly_pattern.pivot(index='sku_id', columns='weekday', values='price_change_flag')
        
        # 计算SKU之间的相关性
        correlation = weekly_pattern.corr()
        
        # 绘制相关性热力图
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
        plt.title('每周价格变动相关性')
        plt.xlabel('星期')
        plt.ylabel('星期')
        plt.xticks(range(7), ['周一', '周二', '周三', '周四', '周五', '周六', '周日'])
        plt.yticks(range(7), ['周一', '周二', '周三', '周四', '周五', '周六', '周日'])
        plt.tight_layout()
        plt.savefig('notebooks/analysis_output/weekly_correlation.png', dpi=200)
        plt.close()
        
        return weekly_pattern
        
    def analyze_monthly_pattern(self) -> pd.DataFrame:
        """
        分析每月价格变动模式
        
        Returns:
            pd.DataFrame: 每月价格变动统计
        """
        # 计算每个SKU的每月价格变动频率
        monthly_pattern = self.df.groupby(['sku_id', 'month'])['price_change_flag'].mean().reset_index()
        monthly_pattern = monthly_pattern.pivot(index='sku_id', columns='month', values='price_change_flag')
        
        # 计算SKU之间的相关性
        correlation = monthly_pattern.corr()
        
        # 绘制相关性热力图
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
        plt.title('每月价格变动相关性')
        plt.xlabel('月份')
        plt.ylabel('月份')
        plt.tight_layout()
        plt.savefig('notebooks/analysis_output/monthly_correlation.png', dpi=200)
        plt.close()
        
        return monthly_pattern
        
    def cluster_skus_by_pattern(self, pattern_type: str = 'weekly', n_clusters: int = 5) -> Dict[int, List[str]]:
        """
        基于价格变动模式对SKU进行聚类
        
        Args:
            pattern_type: 模式类型 ('weekly' 或 'monthly')
            n_clusters: 聚类数量
            
        Returns:
            Dict[int, List[str]]: 聚类结果
        """
        if pattern_type == 'weekly':
            pattern_data = self.analyze_weekly_pattern()
        else:
            pattern_data = self.analyze_monthly_pattern()
            
        # 标准化数据
        scaler = StandardScaler()
        pattern_scaled = scaler.fit_transform(pattern_data)
        
        # 计算距离矩阵
        distance_matrix = pdist(pattern_scaled)
        distance_matrix = squareform(distance_matrix)
        
        # 层次聚类
        Z = linkage(distance_matrix, method='ward')
        
        # 绘制树状图
        plt.figure(figsize=(15, 8))
        dendrogram(Z, truncate_mode='lastp', p=n_clusters)
        plt.title(f'SKU价格变动模式聚类 ({pattern_type})')
        plt.xlabel('SKU')
        plt.ylabel('距离')
        plt.tight_layout()
        plt.savefig(f'notebooks/analysis_output/{pattern_type}_clustering.png', dpi=200)
        plt.close()
        
        # 获取聚类结果
        from scipy.cluster.hierarchy import fcluster
        clusters = fcluster(Z, n_clusters, criterion='maxclust')
        cluster_dict = {i: pattern_data.index[clusters == i].tolist() for i in range(1, n_clusters + 1)}
        
        return cluster_dict
        
    def analyze_cluster_patterns(self, cluster_dict: Dict[int, List[str]], pattern_type: str = 'weekly'):
        """
        分析每个聚类的价格变动模式
        
        Args:
            cluster_dict: 聚类结果
            pattern_type: 模式类型 ('weekly' 或 'monthly')
        """
        for cluster_id, sku_list in cluster_dict.items():
            # 获取该聚类的SKU数据
            cluster_data = self.df[self.df['sku_id'].isin(sku_list)]
            
            # 计算平均价格变动频率
            if pattern_type == 'weekly':
                pattern = cluster_data.groupby('weekday')['price_change_flag'].mean()
                x_labels = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
            else:
                pattern = cluster_data.groupby('month')['price_change_flag'].mean()
                x_labels = [f'{i}月' for i in range(1, 13)]
            
            # 绘制模式图
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(pattern)), pattern.values)
            plt.title(f'聚类 {cluster_id} 的价格变动模式 ({pattern_type})')
            plt.xlabel('时间')
            plt.ylabel('价格变动频率')
            plt.xticks(range(len(pattern)), x_labels)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(f'notebooks/analysis_output/cluster_{cluster_id}_{pattern_type}_pattern.png', dpi=200)
            plt.close()
            
            # 输出统计信息
            logger.info(f"\n聚类 {cluster_id} 统计信息:")
            logger.info(f"SKU数量: {len(sku_list)}")
            logger.info(f"平均价格变动频率: {pattern.mean():.3f}")
            logger.info(f"最高变动频率: {pattern.max():.3f} ({x_labels[pattern.argmax()]})")
            logger.info(f"最低变动频率: {pattern.min():.3f} ({x_labels[pattern.argmin()]})")
            
    def find_similar_skus(self, target_sku: str, pattern_type: str = 'weekly', top_n: int = 5) -> List[Tuple[str, float]]:
        """
        查找与目标SKU价格变动模式相似的SKU
        
        Args:
            target_sku: 目标SKU ID
            pattern_type: 模式类型 ('weekly' 或 'monthly')
            top_n: 返回的相似SKU数量
            
        Returns:
            List[Tuple[str, float]]: 相似SKU列表及其相似度
        """
        if pattern_type == 'weekly':
            pattern_data = self.analyze_weekly_pattern()
        else:
            pattern_data = self.analyze_monthly_pattern()
            
        # 计算目标SKU的模式
        target_pattern = pattern_data.loc[target_sku]
        
        # 计算与其他SKU的相似度
        similarities = []
        for sku in pattern_data.index:
            if sku != target_sku:
                pattern = pattern_data.loc[sku]
                similarity = 1 - np.corrcoef(target_pattern, pattern)[0, 1]
                similarities.append((sku, similarity))
                
        # 按相似度排序
        similarities.sort(key=lambda x: x[1])
        
        return similarities[:top_n]

def main():
    # 创建输出目录
    os.makedirs('notebooks/analysis_output', exist_ok=True)
    
    # 读取数据
    df = pd.read_csv('data/raw/cleaning_data.csv')
    
    # 创建分析器
    analyzer = PricePatternAnalyzer(df)
    
    # 分析每周模式
    logger.info("分析每周价格变动模式...")
    weekly_pattern = analyzer.analyze_weekly_pattern()
    
    # 分析每月模式
    logger.info("分析每月价格变动模式...")
    monthly_pattern = analyzer.analyze_monthly_pattern()
    
    # 基于每周模式聚类
    logger.info("基于每周模式进行SKU聚类...")
    weekly_clusters = analyzer.cluster_skus_by_pattern(pattern_type='weekly', n_clusters=5)
    analyzer.analyze_cluster_patterns(weekly_clusters, pattern_type='weekly')
    
    # 基于每月模式聚类
    logger.info("基于每月模式进行SKU聚类...")
    monthly_clusters = analyzer.cluster_skus_by_pattern(pattern_type='monthly', n_clusters=5)
    analyzer.analyze_cluster_patterns(monthly_clusters, pattern_type='monthly')
    
    # 找出最活跃的SKU
    sku_changes = df.groupby('sku_id')['price_change_flag'].sum().sort_values(ascending=False)
    most_active_sku = sku_changes.index[0]
    
    # 查找相似SKU
    logger.info(f"\n查找与最活跃SKU ({most_active_sku}) 相似的SKU...")
    similar_skus = analyzer.find_similar_skus(most_active_sku, pattern_type='weekly')
    logger.info("相似SKU列表:")
    for sku, similarity in similar_skus:
        logger.info(f"SKU: {sku}, 相似度: {similarity:.3f}")

if __name__ == "__main__":
    main() 