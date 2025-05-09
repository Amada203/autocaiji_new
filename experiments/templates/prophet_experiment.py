"""
Prophet模型实验模板
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.prophet_model import ProphetModel
import json
import os
from datetime import datetime

class ProphetExperiment:
    def __init__(self, config_path, experiment_name=None):
        """
        初始化Prophet实验
        
        参数:
            config_path: 配置文件路径
            experiment_name: 实验名称
        """
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        # 设置实验名称
        if experiment_name is None:
            self.experiment_name = f"prophet_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            self.experiment_name = experiment_name
            
        # 创建实验目录
        self.experiment_dir = os.path.join('experiments/runs', self.experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # 保存配置
        with open(os.path.join(self.experiment_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)
            
        # 初始化模型
        self.model = ProphetModel(**self.config.get('model_params', {}))
        
    def load_data(self):
        """加载数据"""
        data_path = self.config.get('data_path')
        if data_path is None:
            raise ValueError("配置中未指定数据路径")
            
        # 加载数据
        self.data = pd.read_csv(data_path)
        
        # 确保数据格式正确
        if 'ds' not in self.data.columns or 'y' not in self.data.columns:
            raise ValueError("数据必须包含'ds'和'y'列")
            
        # 转换日期格式
        self.data['ds'] = pd.to_datetime(self.data['ds'])
        
        # 划分训练集和测试集
        test_size = self.config.get('test_size', 0.2)
        split_idx = int(len(self.data) * (1 - test_size))
        self.train_data = self.data.iloc[:split_idx]
        self.test_data = self.data.iloc[split_idx:]
        
        print(f"数据加载完成。训练集大小: {len(self.train_data)}，测试集大小: {len(self.test_data)}")
        
    def train(self):
        """训练模型"""
        print("开始训练模型...")
        self.model.fit(self.train_data)
        print("模型训练完成")
        
    def evaluate(self):
        """评估模型"""
        print("开始评估模型...")
        metrics = self.model.evaluate(self.test_data)
        
        # 保存评估结果
        with open(os.path.join(self.experiment_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
            
        print("评估指标:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
            
        return metrics
    
    def visualize(self):
        """可视化结果"""
        print("生成预测可视化...")
        
        # 生成预测
        future = self.model.model.make_future_dataframe(periods=self.config.get('forecast_periods', 30))
        forecast = self.model.model.predict(future)
        
        # 绘制预测结果
        fig = self.model.model.plot(forecast)
        fig.savefig(os.path.join(self.experiment_dir, 'forecast.png'))
        
        # 绘制组件图
        fig = self.model.model.plot_components(forecast)
        fig.savefig(os.path.join(self.experiment_dir, 'components.png'))
        
        print(f"可视化结果已保存到 {self.experiment_dir}")
        
    def save_model(self):
        """保存模型"""
        model_path = os.path.join(self.experiment_dir, 'model.pkl')
        self.model.save(model_path)
        print(f"模型已保存到 {model_path}")
        
    def run(self):
        """运行完整实验流程"""
        self.load_data()
        self.train()
        self.evaluate()
        self.visualize()
        self.save_model()
        print(f"实验 {self.experiment_name} 完成") 