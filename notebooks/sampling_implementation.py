#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
商品价格智能采样策略实现示例
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from scipy.stats import pearsonr
import lightgbm as lgb
from statsmodels.tsa.stattools import grangercausalitytests
import networkx as nx

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']  # macOS优先使用Arial Unicode MS
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建输出目录
os.makedirs('outputs', exist_ok=True)
os.makedirs('outputs/figures', exist_ok=True)

class DataNormalizer:
    """数据标准化和归一化工具类"""
    
    @staticmethod
    def standardize(data: pd.DataFrame, cols: List[str] = None) -> Tuple[pd.DataFrame, StandardScaler]:
        """
        标准化数据 (Z-score标准化)
        
        Args:
            data: 输入数据框
            cols: 需要标准化的列，默认为None表示所有数值列
            
        Returns:
            Tuple[pd.DataFrame, StandardScaler]: 标准化后的数据框和标准化器
        """
        if cols is None:
            cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
            
        scaler = StandardScaler()
        data_copy = data.copy()
        data_copy[cols] = scaler.fit_transform(data_copy[cols])
        
        return data_copy, scaler
    
    @staticmethod
    def normalize(data: pd.DataFrame, cols: List[str] = None) -> Tuple[pd.DataFrame, MinMaxScaler]:
        """
        归一化数据 (Min-Max归一化到[0,1]区间)
        
        Args:
            data: 输入数据框
            cols: 需要归一化的列，默认为None表示所有数值列
            
        Returns:
            Tuple[pd.DataFrame, MinMaxScaler]: 归一化后的数据框和归一化器
        """
        if cols is None:
            cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
            
        scaler = MinMaxScaler()
        data_copy = data.copy()
        data_copy[cols] = scaler.fit_transform(data_copy[cols])
        
        return data_copy, scaler
    
    @staticmethod
    def standardize_and_normalize(data: pd.DataFrame, cols: List[str] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        先标准化再归一化数据
        
        Args:
            data: 输入数据框
            cols: 需要处理的列，默认为None表示所有数值列
            
        Returns:
            Tuple[pd.DataFrame, Dict]: 处理后的数据框和处理器字典
        """
        std_data, std_scaler = DataNormalizer.standardize(data, cols)
        norm_data, norm_scaler = DataNormalizer.normalize(std_data, cols)
        
        scalers = {
            'standard': std_scaler,
            'minmax': norm_scaler
        }
        
        return norm_data, scalers


class DataVisualizer:
    """数据可视化工具类"""
    
    @staticmethod
    def plot_price_trends(df: pd.DataFrame, sku_ids: List[str], output_path: str = None):
        """
        绘制SKU价格趋势图
        
        Args:
            df: 数据框
            sku_ids: 要绘制的SKU ID列表
            output_path: 输出路径，默认为None表示不保存
        """
        plt.figure(figsize=(14, 8))
        
        for sku_id in sku_ids:
            sku_data = df[df['sku_id'] == sku_id].sort_values('record_date')
            plt.plot(sku_data['record_date'], sku_data['price'], label=f'SKU {sku_id}')
            
        plt.title('SKU价格趋势', fontsize=15)
        plt.xlabel('日期')
        plt.ylabel('价格')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"价格趋势图已保存到: {output_path}")
            
        plt.close()
    
    @staticmethod
    def plot_cluster_patterns(clusters: Dict, sku_features: pd.DataFrame, output_path: str = None):
        """
        绘制聚类结果可视化图
        
        Args:
            clusters: 聚类结果
            sku_features: SKU特征数据框
            output_path: 输出路径，默认为None表示不保存
        """
        # 准备数据
        cluster_data = []
        for cluster_id, info in clusters.items():
            for sku_id in info['sku_list']:
                if sku_id in sku_features.index:
                    row = sku_features.loc[sku_id].copy()
                    row['cluster'] = cluster_id
                    cluster_data.append(row)
        
        cluster_df = pd.DataFrame(cluster_data)
        
        # 选择两个主要特征进行可视化 (例如变化频率和价格变异系数)
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 为每个聚类使用不同颜色
        colors = plt.cm.tab10(np.linspace(0, 1, len(clusters)))
        
        for i, (cluster_id, color) in enumerate(zip(clusters.keys(), colors)):
            cluster_points = cluster_df[cluster_df['cluster'] == cluster_id]
            ax.scatter(
                cluster_points['change_freq'], 
                cluster_points['price_cv'],
                s=60, color=color, alpha=0.7,
                label=f'聚类 {cluster_id}: {clusters[cluster_id]["sku_count"]} SKUs'
            )
            
        ax.set_title('SKU聚类结果可视化', fontsize=15)
        ax.set_xlabel('价格变化频率')
        ax.set_ylabel('价格变异系数')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(fontsize=12)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"聚类图已保存到: {output_path}")
            
        plt.close()
    
    @staticmethod
    def plot_weekday_patterns(sku_time_patterns: Dict, output_path: str = None):
        """
        绘制工作日价格变化模式热图
        
        Args:
            sku_time_patterns: SKU时间模式
            output_path: 输出路径，默认为None表示不保存
        """
        # 准备数据
        weekday_data = []
        for sku_id, pattern in sku_time_patterns.items():
            if 'weekday_probs' in pattern:
                weekday_row = {'sku_id': sku_id}
                weekday_row.update({f"周{i}": prob for i, prob in pattern['weekday_probs'].items()})
                weekday_data.append(weekday_row)
        
        if not weekday_data:
            logger.warning("无法绘制工作日模式图：数据为空")
            return
            
        weekday_df = pd.DataFrame(weekday_data)
        weekday_df = weekday_df.set_index('sku_id')
        
        # 选择前30个SKU
        if len(weekday_df) > 30:
            # 选择变化模式最显著的SKU
            weekday_df['variance'] = weekday_df.var(axis=1)
            top_skus = weekday_df.nlargest(30, 'variance').index
            weekday_df = weekday_df.loc[top_skus].drop(columns=['variance'])
        
        # 绘制热图
        plt.figure(figsize=(12, 10))
        weekday_columns = [f"周{i}" for i in range(7)]  # 0=周一，6=周日
        
        sns.heatmap(
            weekday_df[weekday_columns],
            cmap='YlOrRd',
            vmin=0,
            vmax=weekday_df[weekday_columns].values.max(),
            cbar_kws={'label': '价格变化概率'}
        )
        
        plt.title('各SKU工作日价格变化概率热图', fontsize=15)
        plt.ylabel('SKU ID')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"工作日模式热图已保存到: {output_path}")
            
        plt.close()
    
    @staticmethod
    def plot_causality_network(causality_graph: nx.DiGraph, sentinel_skus: List[str], 
                             output_path: str = None, max_nodes: int = 30):
        """
        绘制因果关系网络图
        
        Args:
            causality_graph: 因果关系图
            sentinel_skus: 哨兵SKU列表
            output_path: 输出路径，默认为None表示不保存
            max_nodes: 最大展示节点数
        """
        if causality_graph.number_of_nodes() == 0:
            logger.warning("无法绘制因果网络图：图为空")
            return
            
        # 限制节点数量
        if causality_graph.number_of_nodes() > max_nodes:
            # 保留哨兵SKU和连接最多的其他节点
            non_sentinel_nodes = [n for n in causality_graph.nodes() if n not in sentinel_skus]
            node_degrees = dict(causality_graph.degree(non_sentinel_nodes))
            top_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)
            keep_nodes = sentinel_skus + [n for n, _ in top_nodes[:max_nodes - len(sentinel_skus)]]
            
            causality_graph = causality_graph.subgraph(keep_nodes)
        
        plt.figure(figsize=(16, 12))
        
        # 节点位置
        pos = nx.spring_layout(causality_graph, seed=42)
        
        # 边权重
        edge_weights = [
            causality_graph.get_edge_data(u, v)['weight'] * 3 
            for u, v in causality_graph.edges()
        ]
        
        # 哨兵节点和普通节点使用不同颜色
        node_colors = ['red' if node in sentinel_skus else 'skyblue' for node in causality_graph.nodes()]
        
        # 节点大小
        node_sizes = [
            800 if node in sentinel_skus else 300
            for node in causality_graph.nodes()
        ]
        
        # 绘制网络图
        nx.draw_networkx(
            causality_graph,
            pos=pos,
            with_labels=True,
            node_color=node_colors,
            node_size=node_sizes,
            font_size=10,
            width=edge_weights,
            edge_color='gray',
            arrows=True,
            arrowsize=15,
            alpha=0.8
        )
        
        plt.title('SKU价格变化因果关系网络', fontsize=15)
        
        # 添加图例
        plt.plot([0], [0], 'o', c='red', ms=10, label='哨兵SKU')
        plt.plot([0], [0], 'o', c='skyblue', ms=10, label='普通SKU')
        plt.legend(fontsize=12)
        
        plt.axis('off')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"因果网络图已保存到: {output_path}")
            
        plt.close()
    
    @staticmethod
    def plot_feature_importance(feature_importance: pd.DataFrame, output_path: str = None, top_n: int = 20):
        """
        绘制特征重要性图
        
        Args:
            feature_importance: 特征重要性数据框
            output_path: 输出路径，默认为None表示不保存
            top_n: 显示前N个重要特征
        """
        # 获取前N个重要特征
        top_features = feature_importance.head(top_n).copy()
        
        plt.figure(figsize=(12, 8))
        
        # 绘制条形图
        sns.barplot(x='importance', y='feature', data=top_features, palette='viridis')
        
        plt.title(f'前{top_n}个重要特征', fontsize=15)
        plt.xlabel('重要性')
        plt.ylabel('特征')
        plt.grid(True, linestyle='--', alpha=0.3, axis='x')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"特征重要性图已保存到: {output_path}")
            
        plt.close()
    
    @staticmethod
    def plot_sampling_schedule_heatmap(sampling_schedule: Dict[str, List[datetime]], 
                                     start_date: datetime, days: int = 30,
                                     output_path: str = None, max_skus: int = 50):
        """
        绘制采样计划热图
        
        Args:
            sampling_schedule: 采样计划
            start_date: 开始日期
            days: 天数
            output_path: 输出路径
            max_skus: 最大展示SKU数
        """
        # 准备日期范围
        date_range = [start_date + timedelta(days=i) for i in range(days)]
        date_strs = [d.strftime('%m-%d') for d in date_range]
        
        # 准备采样数据
        sampling_data = []
        for sku_id, dates in sampling_schedule.items():
            # 转换为日期字符串集合，方便查找
            date_strs_set = {d.strftime('%m-%d') for d in dates}
            
            row = {'sku_id': sku_id}
            for date_str in date_strs:
                row[date_str] = 1 if date_str in date_strs_set else 0
                
            sampling_data.append(row)
        
        schedule_df = pd.DataFrame(sampling_data)
        schedule_df = schedule_df.set_index('sku_id')
        
        # 如果SKU太多，只取部分SKU
        if len(schedule_df) > max_skus:
            # 选择采样次数最多的SKU
            schedule_df['total_samples'] = schedule_df.sum(axis=1)
            top_skus = schedule_df.nlargest(max_skus, 'total_samples').index
            schedule_df = schedule_df.loc[top_skus].drop(columns=['total_samples'])
        
        # 绘制热图
        plt.figure(figsize=(16, 12))
        
        # 自定义颜色映射: 白色(无采样)到红色(有采样)
        cmap = LinearSegmentedColormap.from_list('custom_cmap', ['white', '#ff5555'])
        
        sns.heatmap(
            schedule_df[date_strs],
            cmap=cmap,
            cbar=False,
            linewidths=0.5,
            linecolor='lightgray'
        )
        
        plt.title('采样计划热图 (红色=采样日)', fontsize=15)
        plt.xlabel('日期')
        plt.ylabel('SKU ID')
        
        # 调整x轴刻度，每3天显示一次
        plt.xticks(range(0, len(date_strs), 3), [date_strs[i] for i in range(0, len(date_strs), 3)])
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"采样计划热图已保存到: {output_path}")
            
        plt.close()

class EnhancedDataProcessor:
    """增强型数据处理器，用于处理价格数据并提取特征"""
    
    def __init__(self, data_path: str):
        """
        初始化数据处理器
        
        Args:
            data_path: 数据文件路径
        """
        self.data_path = data_path
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        """
        加载数据并进行基础处理
        
        Returns:
            处理后的DataFrame
        """
        logger.info(f"加载数据: {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        
        # 基础处理
        self.df['record_date'] = pd.to_datetime(self.df['record_date'])
        self.df = self.df.sort_values(['sku_id', 'record_date'])
        
        # 计算价格变化特征
        self._calculate_price_changes()
        
        return self.df
    
    def _calculate_price_changes(self):
        """计算与价格变化相关的特征"""
        # 计算前一天价格
        self.df['prev_price'] = self.df.groupby('sku_id')['price'].shift(1)
        
        # 价格变化标志
        self.df['price_change_flag'] = (self.df['price'] != self.df['prev_price']).astype(int)
        
        # 价格变化量
        self.df['price_change_amount'] = self.df['price'] - self.df['prev_price']
        
        # 价格变化比例
        self.df['price_change_ratio'] = self.df['price_change_amount'] / self.df['prev_price']
        
        # 价格变化方向
        self.df['price_change_direction'] = np.sign(self.df['price_change_amount'])
        
        # 添加是否促销字段
        self.df['is_promotion'] = (self.df['price'] > self.df['discount_price']).astype(int)
        
        # 添加时间特征
        self.df['weekday'] = self.df['record_date'].dt.weekday
        self.df['month'] = self.df['record_date'].dt.month
        self.df['day'] = self.df['record_date'].dt.day
        self.df['is_month_start'] = self.df['record_date'].dt.is_month_start
        self.df['is_month_end'] = self.df['record_date'].dt.is_month_end
        self.df['is_weekend'] = self.df['weekday'].isin([5, 6]).astype(int)
        
    def extract_time_patterns(self) -> Dict[str, pd.DataFrame]:
        """
        提取每个SKU的时间模式特征
        
        Returns:
            Dict[str, pd.DataFrame]: SKU ID到时间模式的映射
        """
        if self.df is None:
            self.load_data()
            
        time_patterns = {}
        
        for sku_id, group in self.df.groupby('sku_id'):
            # 计算每个工作日的价格变化概率
            weekday_pattern = group.groupby('weekday')['price_change_flag'].mean()
            
            # 计算每月每一天的价格变化概率
            day_pattern = group.groupby('day')['price_change_flag'].mean()
            
            # 计算每月的价格变化概率
            month_pattern = group.groupby('month')['price_change_flag'].mean()
            
            # 汇总时间模式
            patterns = pd.DataFrame({
                'weekday_pattern': weekday_pattern,
                'day_pattern': day_pattern.reindex(range(1, 32)),
                'month_pattern': month_pattern.reindex(range(1, 13))
            })
            
            time_patterns[sku_id] = patterns
            
        logger.info(f"已提取 {len(time_patterns)} 个SKU的时间模式")
        return time_patterns
    
    def get_change_frequency(self) -> pd.Series:
        """
        计算每个SKU的价格变化频率
        
        Returns:
            pd.Series: SKU ID到价格变化频率的映射
        """
        if self.df is None:
            self.load_data()
            
        change_freq = self.df.groupby('sku_id')['price_change_flag'].mean()
        return change_freq


class SkuClusterer:
    """SKU聚类器，根据价格变化模式对SKU进行分组"""
    
    def __init__(self, data: pd.DataFrame):
        """
        初始化SKU聚类器
        
        Args:
            data: 包含价格和价格变化信息的DataFrame
        """
        self.data = data
        self.sku_features = None
        self.clusters = None
        self.normalized_features = None
        self.scalers = None
        
    def extract_clustering_features(self) -> pd.DataFrame:
        """
        提取用于聚类的特征
        
        Returns:
            pd.DataFrame: 每个SKU的特征矩阵
        """
        # 计算每个SKU的价格变化特征
        features = []
        
        for sku_id, group in self.data.groupby('sku_id'):
            # 基本统计特征
            change_freq = group['price_change_flag'].mean()
            price_std = group['price'].std()
            price_cv = price_std / group['price'].mean()
            
            # 时间模式特征
            weekday_pattern = group.groupby('weekday')['price_change_flag'].mean()
            month_pattern = group.groupby('month')['price_change_flag'].mean().reindex(range(1, 13), fill_value=0)
            
            # 变化幅度特征
            change_amount_mean = group['price_change_amount'].abs().mean()
            change_ratio_mean = group['price_change_ratio'].abs().mean()
            
            # 方向特征
            price_increase_ratio = group[group['price_change_direction'] > 0].shape[0] / max(1, group[group['price_change_flag'] == 1].shape[0])
            
            # 促销特征
            promotion_ratio = group['is_promotion'].mean()
            
            # 合并特征
            sku_feature = {
                'sku_id': sku_id,
                'change_freq': change_freq,
                'price_std': price_std,
                'price_cv': price_cv,
                'change_amount_mean': change_amount_mean,
                'change_ratio_mean': change_ratio_mean,
                'price_increase_ratio': price_increase_ratio,
                'promotion_ratio': promotion_ratio
            }
            
            # 添加时间模式特征
            for i, freq in enumerate(weekday_pattern):
                sku_feature[f'weekday_{i}'] = freq
                
            for i, freq in enumerate(month_pattern):
                sku_feature[f'month_{i+1}'] = freq
                
            features.append(sku_feature)
            
        # 转换为DataFrame
        self.sku_features = pd.DataFrame(features).set_index('sku_id')
        
        # 处理缺失值
        self.sku_features = self.sku_features.fillna(0)
        
        # 进行标准化和归一化处理
        self.normalized_features, self.scalers = DataNormalizer.standardize_and_normalize(self.sku_features)
        
        logger.info(f"已提取 {self.sku_features.shape[1]} 个特征用于SKU聚类，并完成了标准化和归一化处理")
        return self.sku_features
    
    def cluster_skus(self, n_clusters: int = 5) -> Dict:
        """
        使用层次聚类对SKU进行分组
        
        Args:
            n_clusters: 聚类数量
            
        Returns:
            Dict: 聚类结果
        """
        if self.normalized_features is None:
            self.extract_clustering_features()
            
        # 使用已标准化和归一化的特征进行聚类
        logger.info(f"使用层次聚类算法将SKU分为 {n_clusters} 组")
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        labels = clustering.fit_predict(self.normalized_features)
        
        # 保存聚类结果
        self.sku_features['cluster'] = labels
        
        # 生成聚类报告
        cluster_stats = {}
        for cluster_id in range(n_clusters):
            cluster_skus = self.sku_features[self.sku_features['cluster'] == cluster_id].index.tolist()
            avg_change_freq = self.sku_features.loc[cluster_skus, 'change_freq'].mean()
            
            cluster_stats[cluster_id] = {
                'sku_count': len(cluster_skus),
                'avg_change_freq': avg_change_freq,
                'sku_list': cluster_skus
            }
            
            logger.info(f"聚类 {cluster_id}: {len(cluster_skus)} 个SKU, 平均变化频率: {avg_change_freq:.4f}")
            
        self.clusters = cluster_stats
        
        # 生成聚类可视化
        DataVisualizer.plot_cluster_patterns(
            self.clusters, 
            self.sku_features, 
            output_path='outputs/figures/cluster_patterns.png'
        )
        
        return cluster_stats
    
    def detect_anomalies(self, eps: float = 0.5, min_samples: int = 5) -> List[str]:
        """
        使用DBSCAN检测异常变化模式的SKU
        
        Args:
            eps: DBSCAN的eps参数
            min_samples: DBSCAN的min_samples参数
            
        Returns:
            List[str]: 异常SKU列表
        """
        if self.normalized_features is None:
            self.extract_clustering_features()
            
        # 使用已标准化和归一化的特征进行异常检测
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan_labels = dbscan.fit_predict(self.normalized_features)
        
        # 找出异常点(标签为-1)
        anomaly_indices = np.where(dbscan_labels == -1)[0]
        anomaly_skus = self.sku_features.index[anomaly_indices].tolist()
        
        logger.info(f"检测到 {len(anomaly_skus)} 个异常价格变化模式的SKU")
        
        # 如果存在异常SKU，生成异常SKU价格趋势图
        if anomaly_skus and len(anomaly_skus) > 0:
            # 最多选择前5个异常SKU进行可视化
            vis_skus = anomaly_skus[:min(5, len(anomaly_skus))]
            DataVisualizer.plot_price_trends(
                self.data,
                vis_skus,
                output_path='outputs/figures/anomaly_price_trends.png'
            )
        
        return anomaly_skus


class CausalityAnalyzer:
    """因果性分析器，用于发现SKU间的价格变化关系"""
    
    def __init__(self, data: pd.DataFrame, max_skus: int = 100):
        """
        初始化因果性分析器
        
        Args:
            data: 包含价格信息的DataFrame
            max_skus: 最大处理的SKU数量
        """
        self.data = data
        self.max_skus = max_skus
        self.price_series = None
        self.norm_price_series = None  # 存储标准化后的价格时间序列
        self.causality_graph = None
        
    def prepare_time_series(self) -> Dict[str, pd.Series]:
        """
        准备用于因果检验的时间序列数据
        
        Returns:
            Dict[str, pd.Series]: SKU ID到价格时间序列的映射
        """
        # 获取变化频率最高的SKU
        change_freq = self.data.groupby('sku_id')['price_change_flag'].mean()
        top_skus = change_freq.nlargest(self.max_skus).index.tolist()
        
        # 为每个SKU创建价格时间序列
        all_dates = self.data['record_date'].unique()
        all_dates.sort()
        date_index = pd.DatetimeIndex(all_dates)
        
        price_series = {}
        for sku_id in top_skus:
            sku_data = self.data[self.data['sku_id'] == sku_id]
            sku_series = pd.Series(
                index=date_index,
                data=np.nan
            )
            
            # 填充价格数据
            for _, row in sku_data.iterrows():
                sku_series[row['record_date']] = row['price']
                
            # 前向填充缺失值
            sku_series = sku_series.fillna(method='ffill')
            
            # 忽略全部缺失的序列
            if sku_series.isna().sum() < len(sku_series) / 2:
                price_series[sku_id] = sku_series
        
        # 创建标准化的价格序列版本
        self.norm_price_series = {}
        
        # 对每个SKU的价格序列单独标准化
        for sku_id, series in price_series.items():
            # Z-score标准化
            mean_price = series.mean()
            std_price = series.std()
            if std_price > 0:  # 避免除以0
                norm_series = (series - mean_price) / std_price
                self.norm_price_series[sku_id] = norm_series
        
        logger.info(f"准备了 {len(price_series)} 个SKU的价格时间序列用于因果分析")
        self.price_series = price_series
        
        # 可视化部分标准化前后的价格序列
        if price_series:
            # 选择前5个SKU进行可视化
            vis_skus = list(price_series.keys())[:min(5, len(price_series))]
            
            # 原始价格趋势图
            DataVisualizer.plot_price_trends(
                self.data,
                vis_skus,
                output_path='outputs/figures/price_trends_original.png'
            )
            
            # 标准化后的价格趋势图
            plt.figure(figsize=(14, 8))
            for sku_id in vis_skus:
                if sku_id in self.norm_price_series:
                    plt.plot(self.norm_price_series[sku_id].index, 
                             self.norm_price_series[sku_id].values, 
                             label=f'SKU {sku_id}')
            
            plt.title('标准化后的SKU价格趋势', fontsize=15)
            plt.xlabel('日期')
            plt.ylabel('标准化价格 (Z-score)')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig('outputs/figures/price_trends_normalized.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        return price_series
    
    def run_granger_tests(self, max_lag: int = 7, use_normalized: bool = True) -> Dict:
        """
        执行Granger因果检验
        
        Args:
            max_lag: 最大滞后天数
            use_normalized: 是否使用标准化后的价格序列
            
        Returns:
            Dict: 因果检验结果
        """
        if self.price_series is None:
            self.prepare_time_series()
            
        # 选择使用原始或标准化价格序列
        if use_normalized and self.norm_price_series:
            price_data = self.norm_price_series
            logger.info("使用标准化后的价格序列进行Granger因果检验")
        else:
            price_data = self.price_series
            logger.info("使用原始价格序列进行Granger因果检验")
            
        skus = list(price_data.keys())
        n_skus = len(skus)
        
        # 存储因果检验结果
        causality_results = {}
        
        # 对每对SKU执行Granger因果检验
        for i in range(n_skus):
            for j in range(n_skus):
                if i == j:
                    continue
                    
                sku_i = skus[i]
                sku_j = skus[j]
                
                # 准备数据
                series_i = price_data[sku_i]
                series_j = price_data[sku_j]
                
                # 确保两个序列长度相同
                aligned_data = pd.DataFrame({
                    'x': series_i,
                    'y': series_j
                })
                aligned_data = aligned_data.dropna()
                
                if len(aligned_data) <= max_lag + 1:
                    continue
                
                # 执行Granger因果检验
                try:
                    test_result = grangercausalitytests(
                        aligned_data[['y', 'x']], 
                        maxlag=max_lag,
                        verbose=False
                    )
                    
                    # 提取最佳滞后期的p值
                    min_p_value = min([test_result[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1)])
                    best_lag = np.argmin([test_result[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1)]) + 1
                    
                    if min_p_value < 0.05:  # 统计显著性阈值
                        if sku_i not in causality_results:
                            causality_results[sku_i] = []
                            
                        causality_results[sku_i].append({
                            'target_sku': sku_j,
                            'p_value': min_p_value,
                            'lag': best_lag
                        })
                except:
                    # 忽略检验失败的情况
                    pass
                    
        logger.info(f"完成Granger因果检验，发现 {sum(len(v) for v in causality_results.values())} 个显著因果关系")
        return causality_results
    
    def build_causality_graph(self, use_normalized: bool = True) -> nx.DiGraph:
        """
        构建因果关系图
        
        Args:
            use_normalized: 是否使用标准化后的价格序列进行因果检验
            
        Returns:
            nx.DiGraph: 因果关系有向图
        """
        causality_results = self.run_granger_tests(use_normalized=use_normalized)
        
        # 创建有向图
        G = nx.DiGraph()
        
        # 添加节点
        for sku_id in self.price_series.keys():
            G.add_node(sku_id)
            
        # 添加边
        for source_sku, targets in causality_results.items():
            for target_info in targets:
                target_sku = target_info['target_sku']
                p_value = target_info['p_value']
                lag = target_info['lag']
                
                G.add_edge(
                    source_sku, 
                    target_sku, 
                    weight=1-p_value,  # 权重为1减p值，使得更显著的关系权重更高
                    lag=lag
                )
                
        logger.info(f"构建因果图完成，包含 {G.number_of_nodes()} 个节点和 {G.number_of_edges()} 条边")
        self.causality_graph = G
        
        # 生成因果图可视化
        if G.number_of_edges() > 0:
            sentinel_skus = self.identify_sentinel_skus(top_n=10)
            DataVisualizer.plot_causality_network(
                G, 
                sentinel_skus,
                output_path='outputs/figures/causality_network.png'
            )
        
        return G
    
    def identify_sentinel_skus(self, top_n: int = 10) -> List[str]:
        """
        识别哨兵SKU（影响最大的SKU）
        
        Args:
            top_n: 返回的哨兵SKU数量
            
        Returns:
            List[str]: 哨兵SKU列表
        """
        if self.causality_graph is None:
            self.build_causality_graph()
            
        # 计算每个节点的出度中心性
        out_centrality = nx.out_degree_centrality(self.causality_graph)
        
        # 按中心性降序排列
        sorted_centrality = sorted(out_centrality.items(), key=lambda x: x[1], reverse=True)
        
        # 提取top_n个哨兵SKU
        sentinel_skus = [sku for sku, _ in sorted_centrality[:top_n]]
        
        logger.info(f"识别出 {len(sentinel_skus)} 个哨兵SKU")
        
        # 保存哨兵SKU的价格趋势
        if sentinel_skus:
            # 最多可视化前5个哨兵SKU
            vis_sentinel_skus = sentinel_skus[:min(5, len(sentinel_skus))]
            DataVisualizer.plot_price_trends(
                self.data,
                vis_sentinel_skus,
                output_path='outputs/figures/sentinel_price_trends.png'
            )
        
        return sentinel_skus


class PriceChangeProbabilityPredictor:
    """价格变化概率预测器"""
    
    def __init__(self, data: pd.DataFrame):
        """
        初始化预测器
        
        Args:
            data: 包含价格和价格变化信息的DataFrame
        """
        self.data = data
        self.model = None
        self.feature_importance = None
        
    def prepare_features(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        准备训练特征
        
        Returns:
            Tuple[pd.DataFrame, pd.Series]: 特征矩阵和目标变量
        """
        logger.info("准备特征数据用于训练价格变化预测模型")
        
        # 特征工程
        features_df = self.data.copy()
        
        # 添加滞后特征
        for lag in [1, 2, 3, 7, 14, 30]:
            features_df[f'price_change_flag_lag{lag}'] = features_df.groupby('sku_id')['price_change_flag'].shift(lag)
            
        # 添加滑动窗口特征
        for window in [3, 7, 14, 30]:
            features_df[f'price_change_freq_{window}d'] = features_df.groupby('sku_id')['price_change_flag'].rolling(window).mean().reset_index(level=0, drop=True)
            
        # 添加星期几的独热编码
        weekday_dummies = pd.get_dummies(features_df['weekday'], prefix='weekday')
        features_df = pd.concat([features_df, weekday_dummies], axis=1)
        
        # 添加月份特征
        month_dummies = pd.get_dummies(features_df['month'], prefix='month')
        features_df = pd.concat([features_df, month_dummies], axis=1)
        
        # 添加月初月末特征
        features_df['is_month_start'] = features_df['record_date'].dt.is_month_start.astype(int)
        features_df['is_month_end'] = features_df['record_date'].dt.is_month_end.astype(int)
        
        # 丢弃不需要的列
        drop_cols = ['record_date', 'price', 'prev_price', 'price_change_amount', 
                    'price_change_ratio', 'price_change_direction', 'weekday', 'month', 'day', 'discount_price']
        features_df = features_df.drop(columns=drop_cols)
        
        # 丢弃含有缺失值的行
        features_df = features_df.dropna()
        
        # 分离特征和目标
        X = features_df.drop(columns=['sku_id', 'price_change_flag'])
        y = features_df['price_change_flag']
        
        logger.info(f"准备完成，特征维度: {X.shape}, 目标维度: {y.shape}")
        return X, y
    
    def train_model(self) -> lgb.Booster:
        """
        训练LightGBM模型
        
        Returns:
            lgb.Booster: 训练好的模型
        """
        X, y = self.prepare_features()
        
        # 准备LightGBM数据集
        train_data = lgb.Dataset(X, label=y)
        
        # 设置参数
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        
        # 训练模型
        logger.info("开始训练LightGBM模型...")
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            valid_sets=[train_data],
            early_stopping_rounds=10,
            verbose_eval=10
        )
        
        # 获取特征重要性
        self.feature_importance = pd.DataFrame({
            'feature': self.model.feature_name(),
            'importance': self.model.feature_importance()
        }).sort_values('importance', ascending=False)
        
        logger.info(f"模型训练完成，最终AUC: {self.model.best_score['training']['auc']:.4f}")
        return self.model
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        预测价格变化概率
        
        Args:
            features: 特征数据
            
        Returns:
            np.ndarray: 价格变化概率
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
            
        return self.model.predict(features)


class SamplingScheduleGenerator:
    """采样计划生成器"""
    
    def __init__(
        self, 
        data: pd.DataFrame, 
        clusters: Dict, 
        sentinel_skus: List[str], 
        causality_graph: nx.DiGraph
    ):
        """
        初始化采样计划生成器
        
        Args:
            data: 包含价格和价格变化信息的DataFrame
            clusters: SKU聚类结果
            sentinel_skus: 哨兵SKU列表
            causality_graph: 因果关系图
        """
        self.data = data
        self.clusters = clusters
        self.sentinel_skus = sentinel_skus
        self.causality_graph = causality_graph
        self.sku_time_patterns = None
        
    def analyze_time_patterns(self) -> Dict[str, Dict]:
        """
        分析每个SKU的最佳采样时间
        
        Returns:
            Dict[str, Dict]: SKU ID到采样时间模式的映射
        """
        logger.info("分析每个SKU的最佳采样时间...")
        
        # 获取所有SKU列表
        all_skus = self.data['sku_id'].unique()
        
        # 存储每个SKU的时间模式
        sku_time_patterns = {}
        
        for sku_id in all_skus:
            sku_data = self.data[self.data['sku_id'] == sku_id]
            
            # 分析工作日模式
            weekday_probs = sku_data.groupby('weekday')['price_change_flag'].mean()
            best_weekdays = weekday_probs.nlargest(3).index.tolist()
            
            # 分析月内日期模式
            day_probs = sku_data.groupby('day')['price_change_flag'].mean()
            best_days = day_probs.nlargest(5).index.tolist()
            
            # 分析月份模式
            month_probs = sku_data.groupby('month')['price_change_flag'].mean()
            best_months = month_probs.nlargest(3).index.tolist()
            
            # 汇总时间模式
            sku_time_patterns[sku_id] = {
                'best_weekdays': best_weekdays,
                'best_days': best_days,
                'best_months': best_months,
                'weekday_probs': weekday_probs.to_dict(),
                'overall_change_prob': sku_data['price_change_flag'].mean()
            }
            
        self.sku_time_patterns = sku_time_patterns
        logger.info(f"完成 {len(sku_time_patterns)} 个SKU的时间模式分析")
        
        # 生成工作日模式热图
        DataVisualizer.plot_weekday_patterns(
            sku_time_patterns,
            output_path='outputs/figures/weekday_patterns.png'
        )
        
        return sku_time_patterns
        
    def generate_fixed_schedule(self, start_date: datetime, days: int = 30) -> Dict[str, List[datetime]]:
        """
        生成固定采样计划
        
        Args:
            start_date: 开始日期
            days: 计划天数
            
        Returns:
            Dict[str, List[datetime]]: SKU ID到采样时间点的映射
        """
        if self.sku_time_patterns is None:
            self.analyze_time_patterns()
            
        logger.info(f"生成从 {start_date} 开始的 {days} 天固定采样计划")
        
        # 日期范围
        date_range = [start_date + timedelta(days=i) for i in range(days)]
        
        # 存储每个SKU的采样计划
        sampling_schedule = {}
        
        # 遍历所有SKU
        for sku_id, pattern in self.sku_time_patterns.items():
            # 根据聚类结果确定采样频率
            cluster_id = None
            for cid, cluster_info in self.clusters.items():
                if sku_id in cluster_info['sku_list']:
                    cluster_id = cid
                    break
                    
            if cluster_id is None:
                # 默认中频采样
                frequency = 'medium'
            else:
                # 根据平均变化频率决定采样频率
                avg_freq = self.clusters[cluster_id]['avg_change_freq']
                if avg_freq > 0.1:
                    frequency = 'high'
                elif avg_freq > 0.03:
                    frequency = 'medium'
                else:
                    frequency = 'low'
            
            # 确定采样日期
            sample_dates = []
            
            for date in date_range:
                should_sample = False
                weekday = date.weekday()
                day = date.day
                month = date.month
                
                # 哨兵SKU每天采样
                if sku_id in self.sentinel_skus:
                    should_sample = True
                    
                # 高频SKU每天采样
                elif frequency == 'high':
                    should_sample = True
                    
                # 中频SKU在最佳工作日和月初月末采样
                elif frequency == 'medium':
                    if weekday in pattern['best_weekdays'] or day in pattern['best_days']:
                        should_sample = True
                        
                # 低频SKU每周采样两次
                elif frequency == 'low':
                    if weekday in [0, 3]:  # 周一和周四
                        should_sample = True
                        
                if should_sample:
                    sample_dates.append(date)
                    
            sampling_schedule[sku_id] = sample_dates
            
        # 计算总采样次数
        total_samples = sum(len(dates) for dates in sampling_schedule.values())
        avg_samples_per_sku = total_samples / len(sampling_schedule)
        
        logger.info(f"固定采样计划生成完成，总采样次数: {total_samples}, 平均每个SKU采样次数: {avg_samples_per_sku:.2f}")
        
        # 生成采样计划热图
        DataVisualizer.plot_sampling_schedule_heatmap(
            sampling_schedule,
            start_date,
            days,
            output_path='outputs/figures/sampling_schedule.png'
        )
        
        return sampling_schedule
    
    def generate_dynamic_triggers(self) -> Dict[str, List[str]]:
        """
        生成动态触发规则
        
        Returns:
            Dict[str, List[str]]: 触发SKU到被触发SKU列表的映射
        """
        logger.info("生成动态触发规则...")
        
        # 存储触发规则
        trigger_rules = {}
        
        # 基于哨兵SKU和因果图生成触发规则
        for sentinel in self.sentinel_skus:
            if sentinel not in self.causality_graph:
                continue
                
            # 获取哨兵SKU可能影响的其他SKU
            affected_skus = []
            for target in self.causality_graph.successors(sentinel):
                edge_data = self.causality_graph.get_edge_data(sentinel, target)
                weight = edge_data.get('weight', 0)
                
                # 如果关系强度足够，则添加到触发列表
                if weight > 0.7:  # 权重阈值
                    affected_skus.append(target)
                    
            if affected_skus:
                trigger_rules[sentinel] = affected_skus
                
        logger.info(f"生成 {len(trigger_rules)} 条动态触发规则")
        return trigger_rules


def main():
    """主函数"""
    # 创建输出目录
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('outputs/figures', exist_ok=True)
    
    logger.info("========== 开始商品价格智能采样策略分析 ==========")
    
    # 1. 数据加载与处理
    logger.info("1. 数据加载与处理")
    data_processor = EnhancedDataProcessor('data/raw/cleaning_data.csv')
    df = data_processor.load_data()
    
    # 选择一部分SKU的价格趋势可视化
    sample_skus = df['sku_id'].unique()[:min(5, len(df['sku_id'].unique()))]
    DataVisualizer.plot_price_trends(
        df, 
        sample_skus,
        output_path='outputs/figures/sample_price_trends.png'
    )
    
    # 2. SKU聚类分析
    logger.info("2. SKU聚类分析")
    clusterer = SkuClusterer(df)
    cluster_results = clusterer.cluster_skus(n_clusters=5)
    anomaly_skus = clusterer.detect_anomalies()
    
    # 3. 因果性分析
    logger.info("3. 因果性分析")
    causality_analyzer = CausalityAnalyzer(df, max_skus=50)
    # 使用标准化的价格序列进行因果分析
    causality_graph = causality_analyzer.build_causality_graph(use_normalized=True)
    sentinel_skus = causality_analyzer.identify_sentinel_skus(top_n=10)
    
    # 4. 价格变化概率预测
    logger.info("4. 价格变化概率预测")
    predictor = PriceChangeProbabilityPredictor(df)
    model = predictor.train_model()
    
    # 生成特征重要性可视化
    DataVisualizer.plot_feature_importance(
        predictor.feature_importance,
        output_path='outputs/figures/feature_importance.png'
    )
    
    # 5. 生成采样计划
    logger.info("5. 生成采样计划")
    scheduler = SamplingScheduleGenerator(
        df, 
        cluster_results, 
        sentinel_skus, 
        causality_graph
    )
    
    # 生成未来30天的固定采样计划
    start_date = datetime.now()
    fixed_schedule = scheduler.generate_fixed_schedule(start_date, days=30)
    
    # 生成动态触发规则
    trigger_rules = scheduler.generate_dynamic_triggers()
    
    # 6. 保存结果
    logger.info("6. 保存结果")
    # 保存聚类结果
    with open('outputs/cluster_results.txt', 'w') as f:
        for cluster_id, info in cluster_results.items():
            f.write(f"聚类 {cluster_id}: {info['sku_count']} 个SKU, 平均变化频率: {info['avg_change_freq']:.4f}\n")
            f.write(f"SKU列表: {', '.join(info['sku_list'][:10])}... (等)\n\n")
    
    # 保存哨兵SKU列表
    with open('outputs/sentinel_skus.txt', 'w') as f:
        f.write(f"哨兵SKU列表 ({len(sentinel_skus)}):\n")
        for sku in sentinel_skus:
            f.write(f"{sku}\n")
    
    # 保存特征重要性
    predictor.feature_importance.to_csv('outputs/feature_importance.csv', index=False)
    
    # 保存采样计划统计
    with open('outputs/sampling_stats.txt', 'w') as f:
        # 计算每个聚类的平均采样次数
        cluster_samples = {}
        for cluster_id, info in cluster_results.items():
            cluster_skus = info['sku_list']
            sample_counts = [len(fixed_schedule.get(sku, [])) for sku in cluster_skus]
            avg_samples = sum(sample_counts) / len(sample_counts) if sample_counts else 0
            cluster_samples[cluster_id] = avg_samples
            
            f.write(f"聚类 {cluster_id} 平均采样次数: {avg_samples:.2f}\n")
            
        # 总体统计
        total_samples = sum(len(dates) for dates in fixed_schedule.values())
        total_skus = len(fixed_schedule)
        f.write(f"\n总采样次数: {total_samples}\n")
        f.write(f"总SKU数: {total_skus}\n")
        f.write(f"平均每个SKU采样次数: {total_samples/total_skus:.2f}\n")
        
        # 触发规则统计
        f.write(f"\n动态触发规则数量: {len(trigger_rules)}\n")
        for trigger, targets in trigger_rules.items():
            f.write(f"  {trigger} -> {len(targets)} 个SKU\n")
    
    # 创建README文件
    with open('outputs/README.md', 'w') as f:
        f.write("# 商品价格智能采样策略分析结果\n\n")
        f.write("## 分析结果概述\n\n")
        f.write(f"- 分析SKU总数: {len(df['sku_id'].unique())}\n")
        f.write(f"- 时间范围: {df['record_date'].min().strftime('%Y-%m-%d')} 至 {df['record_date'].max().strftime('%Y-%m-%d')}\n")
        f.write(f"- 聚类数量: {len(cluster_results)}\n")
        f.write(f"- 识别出的哨兵SKU数量: {len(sentinel_skus)}\n")
        f.write(f"- 异常SKU数量: {len(anomaly_skus)}\n")
        f.write(f"- 生成的采样规则总数: {total_samples}\n\n")
        
        f.write("## 结果文件清单\n\n")
        f.write("- `cluster_results.txt`: SKU聚类结果详情\n")
        f.write("- `sentinel_skus.txt`: 哨兵SKU列表\n")
        f.write("- `feature_importance.csv`: 价格变化预测模型特征重要性\n")
        f.write("- `sampling_stats.txt`: 采样计划统计\n\n")
        
        f.write("## 可视化图表\n\n")
        f.write("- `figures/sample_price_trends.png`: 样本SKU价格趋势\n")
        f.write("- `figures/price_trends_original.png`: 原始价格趋势\n")
        f.write("- `figures/price_trends_normalized.png`: 标准化后的价格趋势\n")
        f.write("- `figures/anomaly_price_trends.png`: 异常SKU价格趋势\n")
        f.write("- `figures/cluster_patterns.png`: 聚类结果可视化\n")
        f.write("- `figures/weekday_patterns.png`: 工作日价格变化模式热图\n")
        f.write("- `figures/causality_network.png`: 因果关系网络图\n")
        f.write("- `figures/sentinel_price_trends.png`: 哨兵SKU价格趋势\n")
        f.write("- `figures/feature_importance.png`: 特征重要性图\n")
        f.write("- `figures/sampling_schedule.png`: 采样计划热图\n")
    
    logger.info("分析完成，结果已保存到outputs目录")
    logger.info("========== 商品价格智能采样策略分析完成 ==========")

if __name__ == "__main__":
    main() 