# Prophet模型文档

## 概述
Prophet是Facebook开发的时间序列预测库，适用于具有强烈季节性和多个季节性的时间序列数据。本项目中使用Prophet处理日常采集的数据，进行趋势分析和预测。

## 使用说明
```python
from models.prophet_model import ProphetModel

# 初始化模型
model = ProphetModel(
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10,
    holidays_prior_scale=10
)

# 训练模型
model.fit(df)

# 预测未来30天
future = model.create_future_dataframe(periods=30)
forecast = model.predict(future)
```

## 参数调优
Prophet模型主要参数：
- `changepoint_prior_scale`: 控制趋势灵活性 (默认0.05)
- `seasonality_prior_scale`: 控制季节性强度 (默认10)
- `holidays_prior_scale`: 控制节假日效应强度 (默认10)
- `seasonality_mode`: 季节性模式，加法或乘法 ('additive'或'multiplicative')

## 最佳实践
1. 数据至少需要包含两个完整季节周期
2. 对于强季节性数据，增大`seasonality_prior_scale`
3. 对于变化剧烈的趋势，增大`changepoint_prior_scale` 