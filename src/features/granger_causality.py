"""
Granger因果检验模块
用于分析SKU之间的价格变化领先滞后关系
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
import networkx as nx
import logging
from typing import Dict, List, Tuple, Optional, Union

logger = logging.getLogger(__name__)

class SKUCausalNetwork:
    """
    SKU因果网络
    基于Granger因果检验构建SKU间价格变化的领先-滞后关系网络
    """
    
    def __init__(self, max_lags=5, test_method='ssr_chi2test', significance_level=0.05):
        """
        初始化SKU因果网络
        
        Args:
            max_lags: 最大滞后阶数
            test_method: 检验方法，可选'ssr_chi2test', 'ssr_ftest', 'lrtest', 'params_ftest'
            significance_level: 显著性水平
        """
        self.max_lags = max_lags
        self.test_method = test_method
        self.significance_level = significance_level
        self.causality_results = {}
        self.causality_matrix = None
        self.network = None
        self.sentinel_skus = None
        
    def fit(self, price_data: pd.DataFrame, sku_col='sku_id', date_col='ds', price_col='y', 
            filter_stationary=True):
        """
        拟合因果网络
        
        Args:
            price_data: 包含SKU价格时间序列的DataFrame
            sku_col: SKU ID列名
            date_col: 日期列名
            price_col: 价格列名
            filter_stationary: 是否过滤非平稳序列
            
        Returns:
            self
        """
        logger.info("开始构建SKU因果关系网络...")
        
        # 验证输入
        required_cols = [sku_col, date_col, price_col]
        missing_cols = [col for col in required_cols if col not in price_data.columns]
        if missing_cols:
            raise ValueError(f"数据缺少必要的列: {missing_cols}")
            
        # 确保日期格式正确
        data = price_data.copy()
        data[date_col] = pd.to_datetime(data[date_col])
        
        # 透视为宽表格式，便于时间序列分析
        price_pivot = data.pivot(index=date_col, columns=sku_col, values=price_col)
        
        # 过滤掉缺失值过多的SKU
        min_valid_ratio = 0.8
        valid_cols = price_pivot.columns[price_pivot.count() / len(price_pivot) >= min_valid_ratio]
        price_pivot = price_pivot[valid_cols]
        
        # 填充缺失值（使用前值填充，更适合时间序列）
        price_pivot = price_pivot.fillna(method='ffill').fillna(method='bfill')
        
        # 计算价格变化序列（使用差分使序列更平稳）
        if filter_stationary:
            price_changes = price_pivot.diff().dropna()
        else:
            price_changes = price_pivot
            
        # 获取所有SKU ID
        skus = price_changes.columns.tolist()
        n_skus = len(skus)
        
        if n_skus < 2:
            logger.warning("SKU数量不足，无法进行因果检验")
            return self
            
        # 初始化因果关系矩阵
        self.causality_matrix = pd.DataFrame(0, index=skus, columns=skus)
        
        # 对每对SKU进行Granger因果检验
        for i, sku_x in enumerate(skus):
            for j, sku_y in enumerate(skus):
                if i == j:  # 跳过自身
                    continue
                    
                # 提取两个SKU的价格变化序列
                xy_data = price_changes[[sku_x, sku_y]].dropna()
                
                if len(xy_data) <= self.max_lags + 1:
                    logger.warning(f"SKU {sku_x} 和 {sku_y} 的有效数据点不足，跳过检验")
                    continue
                    
                try:
                    # 检验 x 是否引起 y
                    # y ~ y.lag + x.lag
                    test_result = grangercausalitytests(
                        xy_data[[sku_y, sku_x]].values,  # 因变量在前，自变量在后
                        maxlag=self.max_lags,
                        verbose=False
                    )
                    
                    # 存储检验结果
                    key = (sku_x, sku_y)
                    self.causality_results[key] = test_result
                    
                    # 检查是否有任何滞后阶数下的p值小于显著性水平
                    # test_method索引[0][0]获取测试统计量和p值
                    significant_lags = []
                    for lag in range(1, self.max_lags + 1):
                        p_value = test_result[lag][0][self.test_method][1]
                        if p_value < self.significance_level:
                            significant_lags.append((lag, p_value))
                            
                    if significant_lags:
                        # 选择p值最小的滞后阶数作为最佳阶数
                        best_lag, min_p = min(significant_lags, key=lambda x: x[1])
                        self.causality_matrix.loc[sku_x, sku_y] = 1 - min_p  # 置信度作为边权重
                        logger.debug(f"发现因果关系: {sku_x} -> {sku_y}, 滞后={best_lag}, p值={min_p:.4f}")
                        
                except Exception as e:
                    logger.error(f"处理 {sku_x} -> {sku_y} 时出错: {str(e)}")
                    
        # 构建因果网络
        self._build_network()
        
        # 识别关键哨兵SKU
        self._identify_sentinels()
        
        logger.info(f"SKU因果网络构建完成，共发现 {self.causality_matrix.sum().sum()} 个显著因果关系")
        return self
    
    def get_causality_strength(self, sku_x, sku_y):
        """
        获取两个SKU之间的因果强度
        
        Args:
            sku_x: 潜在原因SKU
            sku_y: 潜在结果SKU
            
        Returns:
            因果强度（0-1之间，0表示无因果关系）
        """
        if self.causality_matrix is None:
            raise ValueError("请先调用fit方法构建因果网络")
            
        if sku_x not in self.causality_matrix.index or sku_y not in self.causality_matrix.columns:
            return 0.0
            
        return self.causality_matrix.loc[sku_x, sku_y]
    
    def get_sentinel_skus(self, top_n=None):
        """
        获取哨兵SKU列表
        
        Args:
            top_n: 返回前N个哨兵SKU，None返回全部
            
        Returns:
            哨兵SKU列表，按照重要性降序排列
        """
        if self.sentinel_skus is None:
            raise ValueError("请先调用fit方法构建因果网络")
            
        if top_n is None or top_n >= len(self.sentinel_skus):
            return self.sentinel_skus
        else:
            return self.sentinel_skus[:top_n]
    
    def get_affected_skus(self, sentinel_sku, min_strength=0.3):
        """
        获取受给定哨兵SKU影响的SKU列表
        
        Args:
            sentinel_sku: 哨兵SKU ID
            min_strength: 最小因果强度阈值
            
        Returns:
            受影响SKU的字典 {sku_id: 影响强度}
        """
        if self.causality_matrix is None:
            raise ValueError("请先调用fit方法构建因果网络")
            
        if sentinel_sku not in self.causality_matrix.index:
            return {}
            
        # 获取所有受影响的SKU及强度
        affected = self.causality_matrix.loc[sentinel_sku]
        affected = affected[affected >= min_strength]
        
        return affected.to_dict()
    
    def get_correlation_network(self, min_strength=0.3):
        """
        获取关联网络
        
        Args:
            min_strength: 最小因果强度阈值
            
        Returns:
            字典 {哨兵SKU: [受影响的SKU列表]}
        """
        if self.sentinel_skus is None:
            raise ValueError("请先调用fit方法构建因果网络")
            
        network = {}
        for sentinel in self.sentinel_skus:
            affected = self.get_affected_skus(sentinel, min_strength)
            if affected:
                network[sentinel] = list(affected.keys())
                
        return network
    
    def visualize_network(self, output_path=None, min_strength=0.3):
        """
        可视化因果网络
        
        Args:
            output_path: 输出文件路径，None为不保存
            min_strength: 最小因果强度阈值
        """
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError:
            logger.error("缺少可视化依赖: matplotlib, networkx")
            return
            
        if self.network is None:
            self._build_network()
            
        # 创建筛选后的网络副本
        G_filtered = nx.DiGraph()
        
        # 添加所有节点
        for node in self.network.nodes():
            G_filtered.add_node(node)
            
        # 只添加强度大于阈值的边
        for u, v, data in self.network.edges(data=True):
            if data['weight'] >= min_strength:
                G_filtered.add_edge(u, v, weight=data['weight'])
                
        # 设置节点大小（根据出度中心性）
        out_degree = dict(G_filtered.out_degree())
        node_size = {node: 100 + 500 * deg for node, deg in out_degree.items()}
        
        # 设置节点颜色（哨兵SKU为红色）
        node_color = ['red' if node in self.sentinel_skus else 'skyblue' 
                     for node in G_filtered.nodes()]
        
        # 设置边宽度（根据因果强度）
        edge_width = [data['weight'] * 2 for _, _, data in G_filtered.edges(data=True)]
        
        # 绘制网络
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G_filtered, seed=42)
        
        nx.draw_networkx_nodes(G_filtered, pos, 
                              node_size=[node_size[node] for node in G_filtered.nodes()],
                              node_color=node_color, alpha=0.8)
        
        nx.draw_networkx_edges(G_filtered, pos, width=edge_width, alpha=0.6, 
                              edge_color='gray', arrows=True, arrowsize=15)
        
        nx.draw_networkx_labels(G_filtered, pos, font_size=8)
        
        plt.title("SKU价格变化因果网络")
        plt.axis('off')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"因果网络可视化已保存至 {output_path}")
            
        plt.close()
    
    def _build_network(self):
        """构建网络图"""
        self.network = nx.DiGraph()
        
        # 添加所有SKU作为节点
        for sku in self.causality_matrix.index:
            self.network.add_node(sku)
            
        # 添加因果关系作为有向边
        for sku_x in self.causality_matrix.index:
            for sku_y in self.causality_matrix.columns:
                weight = self.causality_matrix.loc[sku_x, sku_y]
                if weight > 0:
                    self.network.add_edge(sku_x, sku_y, weight=weight)
    
    def _identify_sentinels(self):
        """识别关键哨兵SKU"""
        if self.network is None:
            self._build_network()
            
        # 计算出度中心性（影响其他SKU的程度）
        out_centrality = nx.out_degree_centrality(self.network)
        
        # 计算PageRank（考虑网络整体结构的重要性）
        pagerank = nx.pagerank(self.network)
        
        # 综合考虑两种中心性
        combined_importance = {sku: 0.7 * out_centrality.get(sku, 0) + 0.3 * pagerank.get(sku, 0)
                              for sku in self.network.nodes()}
        
        # 根据重要性排序
        sorted_skus = sorted(combined_importance.items(), key=lambda x: x[1], reverse=True)
        
        # 使用重要性曲线的拐点选择哨兵SKU
        importances = [imp for _, imp in sorted_skus]
        
        if len(importances) <= 1:
            self.sentinel_skus = [sku for sku, _ in sorted_skus]
            return
            
        # 计算重要性的一阶差分
        diffs = np.diff(importances)
        
        # 找到最大差分的位置作为截断点
        if len(diffs) > 0:
            cutoff = np.argmax(diffs) + 1
            cutoff = max(cutoff, min(5, len(sorted_skus)))  # 至少选择5个（如果有的话）
        else:
            cutoff = len(sorted_skus)
            
        self.sentinel_skus = [sku for sku, _ in sorted_skus[:cutoff]]
        logger.info(f"已识别 {len(self.sentinel_skus)} 个关键哨兵SKU") 