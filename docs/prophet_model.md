# Prophet模型文档 (更新)

## 概述
Prophet是Facebook开发的时间序列预测库，适用于具有强烈季节性和多个季节性的时间序列数据。本项目中使用Prophet处理日常采集的数据，进行趋势分析和预测。

## 核心功能
1. 时间序列分解
   - 趋势
   - 季节性
   - 节假日效应
   - 残差

2. 特征生成
   - 趋势特征
   - 季节性特征
   - 变化点检测

3. 预测功能
   - 单变量预测
   - 不确定性区间
   - 自定义季节周期

## 使用说明
```python
from models.prophet_model import ProphetModel

# 初始化模型
model = ProphetModel(
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10,
    holidays_prior_scale=10,
    seasonality_mode='additive'
)

# 训练模型
model.fit(df)

# 预测未来30天
future = model.create_future_dataframe(periods=30)
forecast = model.predict(future)

# 生成特征 (用于混合模型)
features = model.extract_features(df)
```

## 参数调优指南

### 基础参数
| 参数 | 说明 | 推荐值 |
|------|------|-------|
| `changepoint_prior_scale` | 趋势灵活性 | 0.01-0.5 |
| `seasonality_prior_scale` | 季节性强度 | 1-20 |
| `holidays_prior_scale` | 节假日效应强度 | 1-20 |
| `seasonality_mode` | 季节性模式 | 'additive'/'multiplicative' |

### 高级参数
| 参数 | 说明 | 适用场景 |
|------|------|---------|
| `n_changepoints` | 变化点数量 | 趋势变化频繁的数据 |
| `yearly_seasonality` | 年季节性 | 年度数据 |
| `weekly_seasonality` | 周季节性 | 周度数据 |
| `daily_seasonality` | 日季节性 | 日内数据 |

## 与混合模型集成

Prophet模型生成的以下特征可用于混合模型：
1. 趋势分量
2. 季节性分量
3. 变化点标志
4. 预测不确定性

集成示例：
```python
# 在混合模型中
prophet_features = prophet_model.extract_features(X)
X = pd.concat([X, prophet_features], axis=1)
```

## 性能优化
1. 数据预处理
   - 确保时间戳连续性
   - 处理缺失值
   - 异常值检测

2. 计算优化
   - 使用`uncertainty_samples=0`禁用不确定性计算
   - 减少`n_changepoints`数量
   - 禁用不需要的季节性

3. 并行化
   - 多序列并行预测
   - 使用`parallel_backend`参数

## 常见问题
1. 预测值不稳定
   - 检查`changepoint_prior_scale`
   - 增加`changepoint_range`

2. 季节性不明显
   - 增大`seasonality_prior_scale`
   - 检查数据周期

3. 节假日效应过强
   - 减小`holidays_prior_scale`
   - 检查节假日定义