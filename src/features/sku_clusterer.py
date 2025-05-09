"""
SKU聚类器
基于价格变化模式对SKU分组
"""
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union

logger = logging.getLogger(__name__)

class SKUClusterer:
    """
    SKU聚类器
    根据价格变化模式将SKU分为不同组
    """
    
    def __init__(self, method='hierarchical', n_clusters=None, max_clusters=10):
        """
        初始化聚类器
        
        Args:
            method: 聚类方法，'hierarchical'或'dbscan'
            n_clusters: 聚类数量，None时自动确定
            max_clusters: 自动确定聚类数量时的最大聚类数
        """
        self.method = method
        self.n_clusters = n_clusters
        self.max_clusters = max_clusters
        self.clusterer = None
        self.scaler = StandardScaler()
        self.fitted = False
        self.cluster_labels = None
        self.feature_importance = None
        
    def fit(self, data: pd.DataFrame, sku_col='sku_id', features=None, additional_features=None):
        """
        拟合聚类模型
        
        Args:
            data: 包含SKU和价格数据的DataFrame
            sku_col: SKU ID列名
            features: 用于聚类的特征列列表，None时自动提取
            additional_features: 额外的特征列，与自动提取的特征合并
            
        Returns:
            self
        """
        logger.info(f"开始使用{self.method}方法对SKU进行聚类")
        
        # 验证输入
        if sku_col not in data.columns:
            raise ValueError(f"数据必须包含SKU ID列: {sku_col}")
            
        # 提取聚类特征
        if features is None:
            feature_data = self._extract_features(data, sku_col)
            
            # 合并额外特征（如果有）
            if additional_features is not None:
                additional_df = additional_features.copy()
                if sku_col not in additional_df.columns:
                    raise ValueError(f"额外特征DataFrame必须包含{sku_col}列")
                feature_data = pd.merge(feature_data, additional_df, on=sku_col, how='left')
        else:
            # 检查所有指定特征是否存在
            missing_features = [f for f in features if f not in data.columns]
            if missing_features:
                raise ValueError(f"以下特征在数据中不存在: {missing_features}")
                
            # 按SKU聚合特征
            feature_data = data.groupby(sku_col)[features].mean().reset_index()
        
        # 获取SKU列表和特征矩阵
        skus = feature_data[sku_col].values
        
        # 保存原始特征数据（用于后续分析）
        self.feature_data = feature_data.copy()
        
        # 删除非数值特征
        X = feature_data.drop(columns=[sku_col]).select_dtypes(include=['number'])
        
        # 检查是否有足够的数值特征
        if X.shape[1] == 0:
            raise ValueError("没有足够的数值特征进行聚类")
            
        # 记录特征名
        self.feature_names = X.columns.tolist()
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 降维以可视化（不影响聚类）
        if X_scaled.shape[1] > 2:
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X_scaled)
            self.X_2d = X_2d  # 保存用于可视化
            self.pca_components = pca.components_  # 保存主成分
            self.pca_explained_variance = pca.explained_variance_ratio_  # 保存解释方差
        else:
            self.X_2d = X_scaled
            
        # 确定最佳聚类数（如果未指定）
        if self.method == 'hierarchical' and self.n_clusters is None:
            self.n_clusters = self._find_optimal_clusters(X_scaled)
            logger.info(f"自动确定的最佳聚类数: {self.n_clusters}")
        
        # 执行聚类
        if self.method == 'hierarchical':
            self.clusterer = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                linkage='ward'
            )
            cluster_labels = self.clusterer.fit_predict(X_scaled)
        elif self.method == 'dbscan':
            # DBSCAN自动确定聚类数
            self.clusterer = DBSCAN(eps=0.5, min_samples=5)
            cluster_labels = self.clusterer.fit_predict(X_scaled)
            # 将-1（噪声点）重新分配到最近的簇
            if -1 in cluster_labels:
                self._reassign_outliers(X_scaled, cluster_labels)
        else:
            raise ValueError(f"不支持的聚类方法: {self.method}")
            
        # 创建SKU到簇标签的映射
        self.cluster_labels = {sku: label for sku, label in zip(skus, cluster_labels)}
        
        # 计算簇特征重要性
        if hasattr(self, 'feature_names'):
            self._calculate_feature_importance(X, cluster_labels)
            
        # 对每个簇进行特征分析
        self._analyze_clusters(feature_data, cluster_labels, sku_col)
            
        self.fitted = True
        
        # 记录聚类结果统计
        unique_labels = np.unique(cluster_labels)
        cluster_counts = {label: np.sum(cluster_labels == label) for label in unique_labels}
        logger.info(f"聚类完成，共{len(unique_labels)}个簇: {cluster_counts}")
        
        return self
    
    def predict(self, skus):
        """
        预测SKU所属的簇
        
        Args:
            skus: SKU ID或ID列表
            
        Returns:
            簇标签或标签列表
        """
        if not self.fitted:
            raise ValueError("请先调用fit方法进行聚类")
            
        if isinstance(skus, str):
            # 单个SKU
            return self.cluster_labels.get(skus, -1)  # 未知SKU返回-1
        else:
            # SKU列表
            return [self.cluster_labels.get(sku, -1) for sku in skus]
    
    def get_cluster_members(self, cluster_label):
        """
        获取簇中的所有SKU
        
        Args:
            cluster_label: 簇标签
            
        Returns:
            SKU ID列表
        """
        if not self.fitted:
            raise ValueError("请先调用fit方法进行聚类")
            
        return [sku for sku, label in self.cluster_labels.items() if label == cluster_label]
    
    def get_all_clusters(self):
        """
        获取所有聚类结果
        
        Returns:
            dict: {簇标签: [sku_id列表]}
        """
        if not self.fitted:
            raise ValueError("请先调用fit方法进行聚类")
            
        clusters = {}
        for sku, label in self.cluster_labels.items():
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(sku)
            
        return clusters
    
    def get_cluster_characteristics(self):
        """
        获取每个簇的特征特性
        
        Returns:
            dict: {簇标签: 特征特性dict}
        """
        if not self.fitted or not hasattr(self, 'cluster_profiles'):
            raise ValueError("请先调用fit方法进行聚类")
            
        return self.cluster_profiles
    
    def recommend_sampling_strategies(self):
        """
        基于簇特性推荐采样策略
        
        Returns:
            dict: {簇标签: 推荐策略}
        """
        if not self.fitted or not hasattr(self, 'cluster_profiles'):
            raise ValueError("请先调用fit方法进行聚类")
            
        strategies = {}
        
        for cluster_id, profile in self.cluster_profiles.items():
            # 基于簇特性确定策略
            change_rate = profile.get('change_rate', 0)
            avg_interval = profile.get('avg_interval', 0)
            weekday_pattern = np.array([
                profile.get(f'weekday_{i}_change_rate', 0) for i in range(7)
            ])
            
            # 确定策略类型
            if change_rate > 0.3:
                # 高频变化簇 - 每日采样
                strategy = 'daily'
            elif change_rate > 0.1:
                # 中频变化簇
                if np.max(weekday_pattern[5:]) > 0.2:
                    # 周末变化明显
                    strategy = 'weekend_focus'
                else:
                    # 工作日变化
                    strategy = 'weekday_based'
            else:
                # 低频变化簇
                if avg_interval < 14:
                    # 较短间隔
                    strategy = 'interval_based'
                else:
                    # 长间隔低频变化
                    strategy = 'sparse'
                    
            strategies[cluster_id] = strategy
            
        return strategies
    
    def visualize_clusters(self, output_path=None):
        """
        可视化聚类结果
        
        Args:
            output_path: 输出文件路径，None时显示图形
            
        Returns:
            matplotlib.figure.Figure
        """
        if not self.fitted:
            raise ValueError("请先调用fit方法进行聚类")
            
        plt.figure(figsize=(14, 12))
        
        # 散点图
        plt.subplot(2, 2, 1)
        cluster_labels = list(self.cluster_labels.values())
        unique_labels = np.unique(cluster_labels)
        
        for label in unique_labels:
            if label == -1:
                # 黑色表示噪声点
                color = 'k'
                marker = 'x'
            else:
                color = plt.cm.Set1(label % 9)
                marker = 'o'
                
            mask = np.array(list(self.cluster_labels.values())) == label
            indices = np.where(mask)[0]
            
            plt.scatter(
                self.X_2d[indices, 0],
                self.X_2d[indices, 1],
                c=[color],
                marker=marker,
                s=60,
                alpha=0.7,
                label=f'Cluster {label}'
            )
            
        plt.title(f'SKU聚类结果 ({self.method}方法)')
        plt.xlabel('维度 1')
        plt.ylabel('维度 2')
        plt.legend()
        
        # 特征重要性（如果有）
        if self.feature_importance is not None:
            plt.subplot(2, 2, 2)
            
            # 排序并显示前10个特征
            top_features = self.feature_importance.sort_values(ascending=False).head(10)
            sns.barplot(x=top_features.values, y=top_features.index)
            
            plt.title('簇区分的主要特征')
            plt.xlabel('重要性')
            
        # 簇的变化率特性
        if hasattr(self, 'cluster_profiles'):
            plt.subplot(2, 2, 3)
            
            # 提取各簇的变化率
            change_rates = {cluster: profile['change_rate'] 
                           for cluster, profile in self.cluster_profiles.items()}
            
            # 绘制变化率条形图
            clusters = list(change_rates.keys())
            rates = list(change_rates.values())
            
            bars = plt.bar(clusters, rates)
            
            # 添加颜色区分
            for i, bar in enumerate(bars):
                if rates[i] > 0.3:
                    bar.set_color('r')  # 高频变化
                elif rates[i] > 0.1:
                    bar.set_color('g')  # 中频变化
                else:
                    bar.set_color('b')  # 低频变化
                    
            plt.title('各簇的价格变化率')
            plt.xlabel('簇标签')
            plt.ylabel('变化率')
            plt.grid(True, alpha=0.3)
            
            # 周内变化模式
            plt.subplot(2, 2, 4)
            
            # 提取各簇的周内变化模式
            weekday_patterns = {}
            for cluster, profile in self.cluster_profiles.items():
                pattern = [profile.get(f'weekday_{i}_change_rate', 0) for i in range(7)]
                weekday_patterns[f'簇{cluster}'] = pattern
                
            # 绘制热图
            weekday_df = pd.DataFrame(weekday_patterns).T
            weekday_df.columns = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
            
            sns.heatmap(weekday_df, cmap='YlGnBu', annot=True, fmt='.2f')
            plt.title('各簇的周内变化模式')
            
        plt.tight_layout()
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"聚类可视化已保存到 {output_path}")
            
        return plt.gcf()
    
    def save_results(self, output_dir):
        """
        保存聚类结果
        
        Args:
            output_dir: 输出目录
        """
        if not self.fitted:
            raise ValueError("请先调用fit方法进行聚类")
            
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存聚类结果
        clusters = self.get_all_clusters()
        result_df = pd.DataFrame([
            {'cluster': cluster, 'sku_id': sku}
            for cluster, skus in clusters.items()
            for sku in skus
        ])
        
        result_path = os.path.join(output_dir, 'sku_clusters.csv')
        result_df.to_csv(result_path, index=False)
        
        # 保存聚类统计
        stats = pd.DataFrame([
            {'cluster': cluster, 'count': len(skus)}
            for cluster, skus in clusters.items()
        ]).sort_values('count', ascending=False)
        
        stats_path = os.path.join(output_dir, 'cluster_stats.csv')
        stats.to_csv(stats_path, index=False)
        
        # 保存特征重要性（如果有）
        if self.feature_importance is not None:
            importance_path = os.path.join(output_dir, 'feature_importance.csv')
            self.feature_importance.to_csv(importance_path)
            
        # 保存簇特性分析
        if hasattr(self, 'cluster_profiles'):
            profiles_df = pd.DataFrame(self.cluster_profiles).T
            profiles_df.index.name = 'cluster'
            profiles_path = os.path.join(output_dir, 'cluster_profiles.csv')
            profiles_df.to_csv(profiles_path)
            
            # 保存策略推荐
            strategies = self.recommend_sampling_strategies()
            strategies_df = pd.DataFrame({
                'cluster': list(strategies.keys()),
                'recommended_strategy': list(strategies.values())
            })
            strategies_path = os.path.join(output_dir, 'recommended_strategies.csv')
            strategies_df.to_csv(strategies_path, index=False)
            
        # 生成可视化
        viz_path = os.path.join(output_dir, 'cluster_visualization.png')
        self.visualize_clusters(viz_path)
        
        logger.info(f"聚类结果已保存到 {output_dir}")
        
    def _extract_features(self, data, sku_col):
        """
        从原始数据中提取SKU聚类特征
        
        Args:
            data: 原始数据
            sku_col: SKU ID列名
            
        Returns:
            特征DataFrame
        """
        logger.info("提取SKU聚类特征")
        
        # 确保必要的列存在
        required_cols = [sku_col, 'ds', 'y']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"数据必须包含以下列: {required_cols}")
            
        # 确保日期格式正确
        data = data.copy()
        if not pd.api.types.is_datetime64_dtype(data['ds']):
            data['ds'] = pd.to_datetime(data['ds'])
            
        # 计算价格变化（如果没有）
        if 'change_flag' not in data.columns:
            data['prev_y'] = data.groupby(sku_col)['y'].shift(1)
            
            # 使用严格不等判定
            data['change_flag'] = (data['y'] != data['prev_y']).astype(int)
            
        # 按SKU提取特征
        features = []
        
        for sku, group in data.groupby(sku_col):
            # 排序
            group = group.sort_values('ds')
            
            # 基本统计量
            price_mean = group['y'].mean()
            price_std = group['y'].std()
            price_min = group['y'].min()
            price_max = group['y'].max()
            price_range = price_max - price_min
            
            # 变化频率
            change_rate = group['change_flag'].mean()
            
            # 周内变化模式
            weekday_changes = group.groupby(group['ds'].dt.dayofweek)['change_flag'].mean()
            weekday_features = {f'weekday_{i}_change_rate': weekday_changes.get(i, 0) 
                               for i in range(7)}
            
            # 月内变化模式
            month_part = (group['ds'].dt.day - 1) // 10  # 0:上旬, 1:中旬, 2:下旬
            month_part_changes = group.groupby(month_part)['change_flag'].mean()
            month_part_features = {f'month_part_{i}_change_rate': month_part_changes.get(i, 0) 
                                  for i in range(3)}
            
            # 月度变化模式
            month_changes = group.groupby(group['ds'].dt.month)['change_flag'].mean()
            month_features = {f'month_{i}_change_rate': month_changes.get(i, 0)
                            for i in range(1, 13)}
            
            # 价格波动特征
            volatility = price_std / (price_mean + 1e-10)
            
            # 变化幅度
            abs_changes = []
            for i in range(1, len(group)):
                if group.iloc[i]['change_flag'] == 1:
                    curr_price = group.iloc[i]['y']
                    prev_price = group.iloc[i-1]['y']
                    abs_change = abs(curr_price - prev_price)
                    rel_change = abs_change / (prev_price + 1e-10)
                    abs_changes.append(rel_change)
            
            mean_change_ratio = np.mean(abs_changes) if abs_changes else 0
            max_change_ratio = np.max(abs_changes) if abs_changes else 0
            
            # 连续变化和稳定期
            change_runs = self._get_runs(group['change_flag'].values)
            stability_runs = self._get_runs(1 - group['change_flag'].values)
            
            mean_change_run = np.mean(change_runs) if change_runs else 0
            max_change_run = np.max(change_runs) if change_runs else 0
            mean_stability_run = np.mean(stability_runs) if stability_runs else 0
            max_stability_run = np.max(stability_runs) if stability_runs else 0
                
            # 变化间隔
            changes = group[group['change_flag'] == 1]
            if len(changes) > 1:
                change_dates = changes['ds'].sort_values()
                intervals = change_dates.diff().dt.days.dropna()
                mean_interval = intervals.mean() if len(intervals) > 0 else 0
                std_interval = intervals.std() if len(intervals) > 0 else 0
                min_interval = intervals.min() if len(intervals) > 0 else 0
                max_interval = intervals.max() if len(intervals) > 0 else 0
            else:
                mean_interval = 0
                std_interval = 0
                min_interval = 0
                max_interval = 0
                
            # 周期性特征
            time_series = group.set_index('ds')['y']
            
            # 添加类别信息（如果有）
            category_name = None
            change_group = None
            if 'category_name' in group.columns:
                category_name = group['category_name'].iloc[0]
            if 'change_group' in group.columns:
                change_group = group['change_group'].iloc[0]
                
            # 汇总特征
            sku_features = {
                sku_col: sku,
                'price_mean': price_mean,
                'price_std': price_std,
                'price_min': price_min,
                'price_max': price_max,
                'price_range': price_range,
                'change_rate': change_rate,
                'volatility': volatility,
                'mean_change_ratio': mean_change_ratio,
                'max_change_ratio': max_change_ratio,
                'mean_change_run': mean_change_run,
                'max_change_run': max_change_run,
                'mean_stability_run': mean_stability_run,
                'max_stability_run': max_stability_run,
                'mean_interval': mean_interval,
                'std_interval': std_interval,
                'min_interval': min_interval,
                'max_interval': max_interval,
                'category_name': category_name,
                'change_group': change_group,
                **weekday_features,
                **month_part_features,
                **month_features
            }
            
            features.append(sku_features)
            
        # 创建特征DataFrame
        feature_df = pd.DataFrame(features)
        
        # 保存特征名供后续使用
        self.feature_names = [col for col in feature_df.columns if col not in [sku_col, 'category_name', 'change_group']]
        
        return feature_df
    
    def _get_runs(self, arr):
        """计算连续1的长度"""
        runs = []
        current_run = 0
        
        for val in arr:
            if val == 1:
                current_run += 1
            else:
                if current_run > 0:
                    runs.append(current_run)
                    current_run = 0
                    
        if current_run > 0:
            runs.append(current_run)
            
        return runs
    
    def _find_optimal_clusters(self, X):
        """
        使用轮廓系数找到最佳聚类数
        
        Args:
            X: 特征矩阵
            
        Returns:
            最佳聚类数
        """
        # 测试不同聚类数的效果
        range_n_clusters = range(2, min(self.max_clusters + 1, len(X) // 5 + 1))
        silhouette_avg_list = []
        
        for n_clusters in range_n_clusters:
            clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
            cluster_labels = clusterer.fit_predict(X)
            
            # 计算轮廓系数
            try:
                silhouette_avg = silhouette_score(X, cluster_labels)
                silhouette_avg_list.append(silhouette_avg)
                logger.debug(f"聚类数 {n_clusters} 的轮廓系数: {silhouette_avg:.3f}")
            except:
                silhouette_avg_list.append(-1)
                logger.debug(f"聚类数 {n_clusters} 无法计算轮廓系数")
                
        # 如果所有聚类数都无法计算轮廓系数，使用默认值
        if all(s == -1 for s in silhouette_avg_list):
            return min(5, len(X) // 10 + 2)
            
        # 找到轮廓系数最大的聚类数
        best_n_clusters = range_n_clusters[np.argmax(silhouette_avg_list)]
        
        return best_n_clusters
    
    def _reassign_outliers(self, X, labels):
        """
        将DBSCAN识别的离群点重新分配到最近的簇
        
        Args:
            X: 特征矩阵
            labels: 聚类标签
            
        Returns:
            更新的标签
        """
        # 找到离群点
        outlier_indices = np.where(labels == -1)[0]
        
        if len(outlier_indices) == 0:
            return labels
            
        # 计算每个簇的质心
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != -1]
        
        if len(unique_labels) == 0:
            # 所有点都是离群点，使用K-means重新聚类
            from sklearn.cluster import KMeans
            n_clusters = min(5, len(X) // 10 + 2)
            kmeans = KMeans(n_clusters=n_clusters)
            return kmeans.fit_predict(X)
            
        centroids = {}
        for label in unique_labels:
            mask = labels == label
            centroids[label] = X[mask].mean(axis=0)
            
        # 将每个离群点分配到最近的簇
        for idx in outlier_indices:
            distances = [np.linalg.norm(X[idx] - centroids[label]) for label in centroids]
            closest_label = list(centroids.keys())[np.argmin(distances)]
            labels[idx] = closest_label
            
        return labels
    
    def _calculate_feature_importance(self, X, labels):
        """
        计算区分不同簇的特征重要性
        
        Args:
            X: 原始特征矩阵
            labels: 聚类标签
        """
        # 计算每个簇在每个特征上的标准差
        unique_labels = np.unique(labels)
        n_features = X.shape[1]
        
        # 计算总体方差
        total_var = np.var(X, axis=0)
        
        # 计算簇内方差（加权平均）
        within_var = np.zeros(n_features)
        for label in unique_labels:
            mask = labels == label
            cluster_size = np.sum(mask)
            if cluster_size > 0:
                cluster_var = np.var(X[mask], axis=0)
                within_var += (cluster_size / len(labels)) * cluster_var
                
        # 计算特征重要性：1 - 簇内方差/总体方差
        importance = 1 - (within_var / (total_var + 1e-10))
        
        # 创建特征重要性Series
        self.feature_importance = pd.Series(importance, index=self.feature_names)
    
    def _analyze_clusters(self, feature_data, cluster_labels, sku_col):
        """
        分析每个簇的特征特性
        
        Args:
            feature_data: 特征数据
            cluster_labels: 聚类标签
            sku_col: SKU ID列名
        """
        # 添加簇标签到特征数据
        feature_data = feature_data.copy()
        feature_data['cluster'] = [cluster_labels[i] for i in range(len(feature_data))]
        
        # 初始化簇特征字典
        self.cluster_profiles = {}
        
        # 对每个簇提取关键特征
        for cluster in np.unique(cluster_labels):
            cluster_data = feature_data[feature_data['cluster'] == cluster]
            
            # 排除非数值列
            numeric_cols = cluster_data.select_dtypes(include=['number']).columns
            numeric_cols = [col for col in numeric_cols if col not in [sku_col, 'cluster']]
            
            # 计算均值作为簇特征
            profile = {}
            for col in numeric_cols:
                profile[col] = cluster_data[col].mean()
                
            # 记录该簇的SKU数量
            profile['sku_count'] = len(cluster_data)
            
            # 记录该簇的类别分布（如果有）
            if 'category_name' in cluster_data.columns:
                category_counts = cluster_data['category_name'].value_counts()
                main_category = category_counts.index[0] if len(category_counts) > 0 else 'unknown'
                profile['main_category'] = main_category
                profile['category_diversity'] = len(category_counts)
                
            # 记录该簇的变化组分布（如果有）
            if 'change_group' in cluster_data.columns:
                change_group_counts = cluster_data['change_group'].value_counts()
                main_change_group = change_group_counts.index[0] if len(change_group_counts) > 0 else 'unknown'
                profile['main_change_group'] = main_change_group
                
            self.cluster_profiles[cluster] = profile 