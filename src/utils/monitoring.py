from prometheus_client import start_http_server, Gauge
import psutil
import time
import socket
from typing import Dict, Any

# 定义监控指标
PIPELINE_STATUS = Gauge(
    'data_pipeline_status', 
    'Data pipeline execution status (1=success, 0=failure)'
)
PROCESSED_RECORDS = Gauge(
    'data_pipeline_processed_records',
    'Number of records processed in last run'
)
PREDICTIONS_GENERATED = Gauge(
    'data_pipeline_predictions_generated',
    'Number of predictions generated in last run'
)
MEMORY_USAGE = Gauge(
    'data_pipeline_memory_usage_mb',
    'Memory usage in megabytes'
)
CPU_USAGE = Gauge(
    'data_pipeline_cpu_usage_percent',
    'CPU usage percentage'
)

def is_port_available(port):
    """检查端口是否可用"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("localhost", port))
            return True
        except socket.error:
            return False

def start_monitoring_server(start_port=8000, max_retries=5):
    """
    启动监控指标服务器
    自动尝试递增端口直到找到可用端口
    
    参数:
        start_port: 起始端口号
        max_retries: 最大重试次数
        
    返回:
        实际使用的端口号
    """
    port = start_port
    retries = 0
    
    while retries < max_retries:
        if is_port_available(port):
            start_http_server(port)
            return port
        port += 1
        retries += 1
    
    raise OSError(f"无法在端口{start_port}-{port}范围内启动监控服务器")

def update_metrics(stats: Dict[str, Any]):
    """更新监控指标"""
    PIPELINE_STATUS.set(1 if stats.get("success") else 0)
    PROCESSED_RECORDS.set(stats.get("processed_records", 0))
    PREDICTIONS_GENERATED.set(stats.get("predictions_generated", 0))
    MEMORY_USAGE.set(stats.get("memory_usage_mb", 0))
    CPU_USAGE.set(stats.get("cpu_usage", 0))

class ResourceMonitor:
    """资源监控器"""
    def __init__(self):
        self.process = psutil.Process()
        
    def collect_metrics(self) -> Dict[str, float]:
        """收集系统资源指标"""
        return {
            "memory_rss_mb": self.process.memory_info().rss / 1024 / 1024,
            "cpu_percent": self.process.cpu_percent(),
            "thread_count": self.process.num_threads(),
            "disk_io_read": self.process.io_counters().read_bytes,
            "disk_io_write": self.process.io_counters().write_bytes
        }