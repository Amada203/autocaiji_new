from typing import List, Dict, Any, Optional
from datetime import date, datetime
import pandas as pd
import os
import json

class CrawlService:
    def __init__(self):
        """初始化爬取服务"""
        # 数据存储路径
        self.data_dir = os.environ.get("DATA_DIR", "data")
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.results_path = os.path.join(self.data_dir, "crawl_results.json")
        self.price_history_path = os.path.join(self.data_dir, "price_history.json")
        
        # 初始化存储
        if not os.path.exists(self.results_path):
            with open(self.results_path, 'w') as f:
                json.dump([], f)
        
        if not os.path.exists(self.price_history_path):
            with open(self.price_history_path, 'w') as f:
                json.dump({}, f)
    
    def process_crawl_results(self, results, crawl_date=None):
        """处理爬取结果
        
        Args:
            results: 爬取结果列表
            crawl_date: 爬取日期
            
        Returns:
            dict: 处理结果
        """
        if crawl_date is None:
            crawl_date = date.today()
            
        # 转换结果为标准格式
        results_data = []
        
        for result in results:
            timestamp = result.timestamp if result.timestamp else datetime.now()
            
            # 标准化数据
            result_data = {
                'sku_id': result.sku_id,
                'price': float(result.price),
                'timestamp': timestamp.isoformat(),
                'crawl_date': crawl_date.isoformat() if isinstance(crawl_date, date) else crawl_date
            }
            
            results_data.append(result_data)
        
        # 保存结果
        self._save_results(results_data)
        
        # 更新价格历史
        self._update_price_history(results_data)
        
        return {
            'status': 'success',
            'processed_count': len(results)
        }
    
    def get_crawl_stats(self, start_date=None, end_date=None):
        """获取爬取统计信息
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            dict: 统计信息
        """
        results = self._load_results()
        
        # 过滤日期范围
        filtered_results = results
        if start_date:
            start_date_str = start_date.isoformat()
            filtered_results = [r for r in filtered_results if r['crawl_date'] >= start_date_str]
        if end_date:
            end_date_str = end_date.isoformat()
            filtered_results = [r for r in filtered_results if r['crawl_date'] <= end_date_str]
        
        # 计算统计信息
        stats = {
            'total_crawls': len(filtered_results),
            'unique_skus': len(set(r['sku_id'] for r in filtered_results)),
            'date_range': {
                'start': min([r['crawl_date'] for r in filtered_results]) if filtered_results else None,
                'end': max([r['crawl_date'] for r in filtered_results]) if filtered_results else None
            },
            'daily_stats': self._calculate_daily_stats(filtered_results)
        }
        
        return stats
    
    def _calculate_daily_stats(self, results):
        """计算每日统计信息"""
        # 按日期分组
        date_groups = {}
        for result in results:
            date = result['crawl_date']
            if date not in date_groups:
                date_groups[date] = []
            date_groups[date].append(result)
        
        # 计算每日统计
        daily_stats = []
        for date, group in date_groups.items():
            daily_stats.append({
                'date': date,
                'crawl_count': len(group),
                'unique_skus': len(set(r['sku_id'] for r in group))
            })
        
        # 按日期排序
        daily_stats.sort(key=lambda x: x['date'], reverse=True)
        
        return daily_stats
    
    def _save_results(self, results_data):
        """保存爬取结果"""
        # 加载现有结果
        existing_results = self._load_results()
        
        # 添加新结果
        all_results = existing_results + results_data
        
        # 保存到文件
        with open(self.results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
    
    def _load_results(self):
        """加载爬取结果"""
        try:
            with open(self.results_path, 'r') as f:
                return json.load(f)
        except:
            return []
    
    def _update_price_history(self, results_data):
        """更新价格历史"""
        try:
            # 加载现有价格历史
            with open(self.price_history_path, 'r') as f:
                price_history = json.load(f)
        except:
            price_history = {}
        
        # 更新价格历史
        for result in results_data:
            sku_id = result['sku_id']
            if sku_id not in price_history:
                price_history[sku_id] = []
            
            # 添加新价格记录
            price_history[sku_id].append({
                'date': result['crawl_date'],
                'price': result['price'],
                'timestamp': result['timestamp']
            })
        
        # 保存到文件
        with open(self.price_history_path, 'w') as f:
            json.dump(price_history, f, indent=2)