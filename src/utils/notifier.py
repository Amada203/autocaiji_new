import smtplib
from email.mime.text import MIMEText
import json
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class EmailNotifier:
    def __init__(self, config: dict):
        self.enabled = config.get("enabled", False)
        if not self.enabled:
            return
            
        self.smtp_server = config["smtp_server"]
        self.smtp_port = config["smtp_port"]
        self.username = config["username"]
        self.password = config["password"]
        self.from_addr = config["from_addr"]
        self.to_addrs = config["to_addrs"]

    def send(self, subject: str, message: str) -> bool:
        """发送邮件通知"""
        if not self.enabled:
            return False
            
        try:
            msg = MIMEText(message)
            msg["Subject"] = subject
            msg["From"] = self.from_addr
            msg["To"] = ", ".join(self.to_addrs)
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.sendmail(self.from_addr, self.to_addrs, msg.as_string())
            
            logger.info("邮件通知发送成功")
            return True
        except Exception as e:
            logger.error(f"邮件通知发送失败: {str(e)}")
            return False

def send_pipeline_notification(success: bool, stats: dict):
    """发送管道运行结果通知"""
    try:
        with open("config/database.json") as f:
            config = json.load(f).get("notification", {}).get("email", {})
            
        notifier = EmailNotifier(config)
        if not notifier.enabled:
            return
            
        status = "成功" if success else "失败"
        subject = f"数据管道运行{status}"
        
        message = f"""
数据管道运行结果:
- 状态: {status}
- 开始时间: {stats.get('start_time')}
- 结束时间: {stats.get('end_time')}
- 处理记录数: {stats.get('processed_records', 0)}
- 生成预测数: {stats.get('predictions_generated', 0)}
- 内存使用: {stats.get('memory_usage_mb', 0):.2f} MB
- CPU使用率: {stats.get('cpu_usage', 0)}%
        """
        
        if not success:
            message += f"\n错误信息: {stats.get('errors', ['未知错误'])[0]}"
            
        notifier.send(subject, message.strip())
    except Exception as e:
        logger.error(f"发送通知时出错: {str(e)}")