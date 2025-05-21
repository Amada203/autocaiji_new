# 模型实现文档 (更新)

## 1. 模型架构概览

### 1.1 基础模型
- `ProphetModel`: 基础时间序列预测
- `ARIMAModel`: 传统时间序列模型
- `LSTMModel`: 深度学习时间序列模型

### 1.2 混合模型
- `PriceModel`: 价格预测混合模型 (Prophet + LightGBM)
- `PriceChangeProbabilityModel`: 价格变化检测模型 (Prophet + LightGBM)
- `FusionModel`: 多模型融合预测

## 2. 详细数据流程

### 2.1 数据获取
- 从Impala和MySQL数据源获取原始数据
- 数据验证和完整性检查
- 自动重试机制

### 2.2 数据预处理
1. 数据清洗
   - 缺失值处理
   - 异常值检测
   - 数据标准化

2. 特征工程
   - 时间特征提取
   - Prophet特征生成
   - 价格波动特征
   - SKU聚类特征

### 2.3 模型训练流程
1. 单模型训练
   - 参数配置
   - 交叉验证
   - 早停机制

2. 混合模型训练
   - Prophet模型训练
   - LightGBM模型训练
   - 特征重要性分析

3. 模型融合
   - 加权平均
   - 堆叠集成
   - 模型选择

## 3. 核心模型实现

### 3.1 PriceChangeProbabilityModel
- 用途: 检测价格变化的概率
- 架构:
  ```mermaid
  graph LR
    A[原始数据] --> B[Prophet特征提取]
    A --> C[时间特征]
    B --> D[特征合并]
    C --> D
    D --> E[LightGBM分类]
    E --> F[概率预测]
  ```
- 关键参数:
  - `change_threshold`: 价格变化判定阈值
  - `prophet_params`: Prophet模型参数
  - `lgbm_params`: LightGBM模型参数

### 3.2 FusionModel
- 用途: 多模型融合预测
- 支持模型:
  - Prophet
  - LightGBM
  - LSTM
- 融合策略:
  - 简单平均
  - 动态加权
  - 模型选择器

## 4. 接口规范

所有模型实现统一的`BaseModel`接口:

```python
class BaseModel:
    def fit(self, X, y=None): ...
    def predict(self, X): ...
    def predict_probability(self, X): ...
    def evaluate(self, X, y): ...
    def save(self, path): ...
    def load(self, path): ...
```

## 5. 模型评估指标

1. 回归指标:
   - MAE, MSE, RMSE
   - R-squared
   - MAPE

2. 分类指标:
   - AUC-ROC
   - Precision-Recall
   - F1 Score
   - 捕获率

## 6. 部署流程

1. 模型训练
2. 模型验证
3. 模型打包
4. API部署
5. 监控告警

## 7. 最佳实践

1. 数据质量:
   - 确保数据完整性
   - 监控数据分布变化

2. 模型更新:
   - 定期重新训练
   - 版本控制
   - A/B测试

3. 性能优化:
   - 特征选择
   - 参数调优
   - 硬件加速