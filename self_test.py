from src.data.data_fetcher import DataFetcher
from datetime import datetime, timedelta
fetcher = DataFetcher()
data = fetcher.fetch_training_data(train_end=datetime.now() - timedelta(days=30), 
                                  val_end=datetime.now() - timedelta(days=15), 
                                  test_end=datetime.now())
print(data.head())  # 检查数据是否非空且格式正确