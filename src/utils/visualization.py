import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import platform
import os
import matplotlib.dates as mdates

class DataVisualizer:
    def __init__(self, df: pd.DataFrame):
        """
        初始化数据可视化器
        
        Args:
            df: 包含价格数据的DataFrame
        """
        self.df = df.copy()
        self._setup_plot_style()
        self._preprocess_data()
        
    def _setup_plot_style(self):
        """设置绘图样式"""
        # 设置中文字体
        if platform.system() == 'Darwin':
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'PingFang HK']
        else:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
    def _preprocess_data(self):
        """数据预处理"""
        # 确保日期格式正确
        self.df['record_date'] = pd.to_datetime(self.df['record_date'])
        
        # 对原始价格和折扣价分别做标准化+归一化处理
        scaler_std = StandardScaler()
        scaler_minmax = MinMaxScaler()
        
        # 原始价格处理
        price_std = scaler_std.fit_transform(self.df[['price']])
        self.df['price_scaled'] = scaler_minmax.fit_transform(price_std)
        
        # 折扣价处理
        discount_price_std = scaler_std.fit_transform(self.df[['discount_price']])
        self.df['discount_price_scaled'] = scaler_minmax.fit_transform(discount_price_std)
        
        # 计算折扣率（并裁剪到0~1）
        self.df['discount_rate'] = (1 - self.df['discount_price'] / self.df['price']) * 100  # 转换为百分比
        self.df['discount_rate'] = self.df['discount_rate'].clip(0, 100)
        
        # 计算价格变动
        self.df = self.df.sort_values(['sku_id', 'record_date'])
        self.df['prev_price'] = self.df.groupby('sku_id')['price'].shift(1)
        self.df['price_change_flag'] = (self.df['price'] != self.df['prev_price']).astype(int)
        self.df['price_change_amount'] = self.df['price'] - self.df['prev_price']
        self.df['price_change_ratio'] = self.df['price_change_amount'] / self.df['prev_price']
        self.df['price_change_direction'] = np.sign(self.df['price_change_amount'])
        
        # 计算连续变动天数
        self.df['price_change_streak'] = self.df.groupby('sku_id')['price_change_flag'].transform(
            lambda x: x.groupby((x != x.shift()).cumsum()).cumsum()
        )
        
    def plot_price_trend(self, 
                        sku_id: str, 
                        category_name: Optional[str] = None,
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> plt.Figure:
        """
        绘制单个SKU的价格趋势图
        
        Args:
            sku_id: SKU ID
            category_name: 类别名称
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            plt.Figure: 图表对象
        """
        # 筛选数据
        mask = self.df['sku_id'] == sku_id
        if category_name:
            mask &= self.df['category_name'] == category_name
        if start_date:
            mask &= self.df['record_date'] >= pd.to_datetime(start_date)
        if end_date:
            mask &= self.df['record_date'] <= pd.to_datetime(end_date)
            
        sku_df = self.df[mask].copy()
        sku_df = sku_df.sort_values('record_date')
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[2, 1])
        
        # 绘制价格
        ax1.plot(sku_df['record_date'], sku_df['price_scaled'], 
                color='green', linewidth=1.2, label='原始价格')
        ax1.plot(sku_df['record_date'], sku_df['discount_price_scaled'], 
                color='blue', linewidth=1.2, linestyle=':', label='折扣后到手价')
        
        # 标记价格变动点
        price_changes = sku_df[sku_df['price_change_flag'] == 1]
        ax1.scatter(price_changes['record_date'], price_changes['price_scaled'],
                   color='red', s=50, label='价格变动点')
        
        ax1.set_ylabel('价格', fontsize=10)
        ax1.set_ylim(0, 1)
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        # 绘制价格变动标志
        ax2.plot(sku_df['record_date'], sku_df['price_change_flag'],
                color='red', linewidth=1.2, label='价格变动标志')
        ax2.set_ylabel('价格变动', fontsize=10)
        ax2.set_ylim(-0.1, 1.1)
        ax2.grid(True, linestyle='--', alpha=0.5)
        
        # 设置标题和网格
        title = f'SKU {sku_id} 价格趋势'
        if category_name:
            title = f'{category_name} - {title}'
        ax1.set_title(title, fontsize=12, fontweight='bold')
        
        # 设置X轴格式
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # 添加图例
        ax1.legend(loc='upper right', fontsize=9)
        ax2.legend(loc='upper right', fontsize=9)
        
        plt.tight_layout()
        return fig
        
    def plot_price_change_distribution(self, output_dir: Optional[str] = None):
        """
        绘制价格变动分布图
        
        Args:
            output_dir: 输出目录
        """
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 价格变动频率分布
        sns.histplot(data=self.df.groupby('sku_id')['price_change_flag'].mean(),
                    ax=axes[0, 0], bins=30)
        axes[0, 0].set_title('价格变动频率分布')
        axes[0, 0].set_xlabel('变动频率')
        axes[0, 0].set_ylabel('SKU数量')
        
        # 价格变动金额分布
        sns.histplot(data=self.df['price_change_amount'].dropna(),
                    ax=axes[0, 1], bins=30)
        axes[0, 1].set_title('价格变动金额分布')
        axes[0, 1].set_xlabel('变动金额')
        axes[0, 1].set_ylabel('频次')
        
        # 价格变动比例分布
        sns.histplot(data=self.df['price_change_ratio'].dropna(),
                    ax=axes[1, 0], bins=30)
        axes[1, 0].set_title('价格变动比例分布')
        axes[1, 0].set_xlabel('变动比例')
        axes[1, 0].set_ylabel('频次')
        
        # 连续变动天数分布
        sns.histplot(data=self.df['price_change_streak'],
                    ax=axes[1, 1], bins=30)
        axes[1, 1].set_title('连续变动天数分布')
        axes[1, 1].set_xlabel('连续变动天数')
        axes[1, 1].set_ylabel('频次')
        
        plt.tight_layout()
        
        # 保存图表
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, 'price_change_distribution.png')
            plt.savefig(output_path, dpi=200)
            plt.close()
            
    def plot_price_change_pattern(self, top_n: int = 5, output_dir: Optional[str] = None):
        """
        绘制价格变动模式分析图
        
        Args:
            top_n: 展示前N个最活跃的SKU
            output_dir: 输出目录
        """
        # 计算每个SKU的价格变动次数
        sku_changes = self.df.groupby('sku_id')['price_change_flag'].sum().sort_values(ascending=False)
        top_skus = sku_changes.head(top_n).index
        
        # 创建图表
        fig, axes = plt.subplots(top_n, 1, figsize=(15, 4 * top_n))
        if top_n == 1:
            axes = [axes]
            
        for i, sku in enumerate(top_skus):
            sku_df = self.df[self.df['sku_id'] == sku].copy()
            sku_df = sku_df.sort_values('record_date')
            
            # 绘制价格
            axes[i].plot(sku_df['record_date'], sku_df['price_scaled'],
                        color='blue', linewidth=1.2, label='价格')
            
            # 标记价格变动点
            price_changes = sku_df[sku_df['price_change_flag'] == 1]
            axes[i].scatter(price_changes['record_date'], price_changes['price_scaled'],
                          color='red', s=50, label='价格变动点')
            
            # 设置标题和网格
            axes[i].set_title(f'SKU {sku} 价格变动模式', fontsize=10, fontweight='bold')
            axes[i].grid(True, linestyle='--', alpha=0.5)
            axes[i].legend(loc='upper right', fontsize=8)
            
            # 设置Y轴范围
            axes[i].set_ylim(0, 1)
            
            # 设置X轴格式
            axes[i].xaxis.set_major_locator(mdates.MonthLocator())
            axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.setp(axes[i].xaxis.get_majorticklabels(), rotation=45)
            
        plt.suptitle(f'Top {top_n} 最活跃SKU价格变动模式', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        
        # 保存图表
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, 'price_change_pattern.png')
            plt.savefig(output_path, dpi=200)
            plt.close()
            
    def plot_weekly_pattern(self, output_dir: Optional[str] = None):
        """
        绘制每周价格变动模式
        
        Args:
            output_dir: 输出目录
        """
        # 计算每周各天的价格变动频率
        self.df['weekday'] = self.df['record_date'].dt.weekday
        weekly_pattern = self.df.groupby('weekday')['price_change_flag'].mean().reset_index()
        
        # 创建图表
        plt.figure(figsize=(10, 6))
        sns.barplot(data=weekly_pattern, x='weekday', y='price_change_flag')
        
        # 设置标题和标签
        plt.title('每周价格变动模式', fontsize=12, fontweight='bold')
        plt.xlabel('星期')
        plt.ylabel('价格变动频率')
        
        # 设置X轴刻度
        plt.xticks(range(7), ['周一', '周二', '周三', '周四', '周五', '周六', '周日'])
        
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        # 保存图表
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, 'weekly_pattern.png')
            plt.savefig(output_path, dpi=200)
            plt.close()
            
    def plot_category_preview(self, 
                            category_name: str,
                            n_sku: int = 20,
                            output_dir: Optional[str] = None):
        """
        绘制某个类别下多个SKU的价格趋势预览
        
        Args:
            category_name: 类别名称
            n_sku: 要预览的SKU数量
            output_dir: 输出目录
        """
        # 获取该类别下的SKU列表
        sku_list = self.df[self.df['category_name'] == category_name]['sku_id'].unique()[:n_sku]
        
        # 创建图表
        fig, axes = plt.subplots(len(sku_list), 1, figsize=(16, 2 * len(sku_list)), sharex=True)
        if len(sku_list) == 1:
            axes = [axes]
            
        for i, sku in enumerate(sku_list):
            sku_df = self.df[(self.df['category_name'] == category_name) & 
                           (self.df['sku_id'] == sku)].copy()
            sku_df = sku_df.sort_values('record_date')
            
            ax1 = axes[i]
            ax2 = ax1.twinx()
            
            # 绘制价格
            ax1.plot(sku_df['record_date'], sku_df['price_scaled'], 
                    color='green', linewidth=1.2, label='原始价格')
            ax1.plot(sku_df['record_date'], sku_df['discount_price_scaled'], 
                    color='blue', linewidth=1.2, linestyle=':', label='折扣后到手价')
            ax1.set_ylabel('价格', fontsize=9)
            
            # 绘制折扣率
            ax2.plot(sku_df['record_date'], sku_df['discount_rate'], 
                    color='orange', linewidth=1.2, linestyle='--', label='折扣率')
            ax2.set_ylabel('折扣率 (%)', fontsize=9)
            
            # 设置Y轴范围
            ax1.set_ylim(0, 1)
            ax2.set_ylim(0, 100)
            
            # 添加图例
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
            
            # 设置标题和网格
            ax1.set_title(f'{category_name} - SKU {sku}', fontsize=10, fontweight='bold')
            ax1.grid(True, linestyle='--', alpha=0.5)
            
            # 设置刻度标签大小
            ax1.tick_params(axis='y', labelsize=8)
            ax2.tick_params(axis='y', labelsize=8)
            
        # 设置X轴格式
        plt.xlabel('月份', fontsize=11)
        plt.suptitle(f'{category_name}品类下{n_sku}个SKU价格与折扣率时序（标准化+归一化）', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        
        # 保存图表
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            safe_cat = category_name.replace('/', '_').replace('\\', '_')
            output_path = os.path.join(output_dir, f'{safe_cat}_sku_price_discount_timeseries_preview.png')
            plt.savefig(output_path, dpi=200)
            plt.close()
            
    def plot_stat_boxplot(self, output_dir: Optional[str] = None):
        """
        绘制SKU统计数据的箱线图
        
        Args:
            output_dir: 输出目录
        """
        # 分组计算CV、标准差、促销频率
        grouped = self.df.groupby(['category_name', 'sku_id'])
        cv = grouped['price'].std() / grouped['price'].mean()
        std = grouped['price'].std()
        promo_freq = grouped['is_promotion'].mean()
        
        res = pd.DataFrame({
            'cv': cv,
            'std': std,
            'promo_freq': promo_freq
        }).reset_index()
        
        # 创建图表
        plt.figure(figsize=(15, 4))
        
        # CV分布
        plt.subplot(1, 3, 1)
        sns.boxplot(y=res['cv'])
        plt.title('全量SKU CV分布')
        plt.ylabel('CV')
        
        # 标准差分布
        plt.subplot(1, 3, 2)
        sns.boxplot(y=res['std'])
        plt.title('全量SKU 标准差分布')
        plt.ylabel('标准差')
        
        # 促销频率分布
        plt.subplot(1, 3, 3)
        sns.boxplot(y=res['promo_freq'])
        plt.title('全量SKU 促销频率分布')
        plt.ylabel('促销频率')
        
        plt.tight_layout()
        
        # 保存图表
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, 'sku_stat_boxplot.png')
            plt.savefig(output_path, dpi=200)
            plt.close()
            
        # 输出统计信息
        print('--- CV分组统计 ---')
        # 分位数法分组
        bins = [-float('inf'), 0.0293, 0.0714, 0.1313, float('inf')]
        labels = ['极低波动', '低波动', '中波动', '高波动']
        res['cv_group'] = pd.cut(res['cv'], bins=bins, labels=labels, right=False)
        
        group_stats = res.groupby('cv_group').agg({
            'sku_id': 'count',
            'cv': ['mean', 'std'],
            'std': ['mean', 'std'],
            'promo_freq': ['mean', 'std']
        })
        print(group_stats)
        print('\n每组SKU数量:')
        print(res['cv_group'].value_counts())
        
    def save_plots(self, output_dir: str):
        """
        保存所有可视化图表
        
        Args:
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取前3个重要品类
        top_categories = self.df['category_name'].value_counts().index[:3].tolist()
        print(f"自动选择的3个重要品类: {top_categories}")
        
        # 为每个品类生成预览图
        for cat in top_categories:
            self.plot_category_preview(cat, n_sku=20, output_dir=output_dir)
            
        # 生成统计箱线图
        self.plot_stat_boxplot(output_dir)
        
        # 生成价格变动分布图
        self.plot_price_change_distribution(output_dir)
        
        # 生成价格变动模式分析图
        self.plot_price_change_pattern(top_n=5, output_dir=output_dir)
        
        # 生成每周价格变动模式图
        self.plot_weekly_pattern(output_dir) 