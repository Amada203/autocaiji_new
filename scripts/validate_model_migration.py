"""
模型迁移验证主脚本
"""
import sys
import os
import logging

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.validation.run_validation import run_full_validation

def main():
    """执行模型迁移验证"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("开始执行模型迁移验证...")
    
    try:
        results = run_full_validation()
        logger.info(f"验证完成，报告已保存至 reports/validation_report.json")
        
        # 输出关键指标
        for metric, data in results['migration_validation'].items():
            if data['significant'] and data['improvement'] > 0:
                logger.info(f"✅ {metric}: 显著提升 {data['improvement']:.2%}")
            elif data['significant'] and data['improvement'] < 0:
                logger.warning(f"⚠️ {metric}: 显著下降 {data['improvement']:.2%}")
            else:
                logger.info(f"ℹ️ {metric}: 变化不显著 {data['improvement']:.2%}")
                
    except Exception as e:
        logger.error(f"验证过程出错: {str(e)}")
        raise

if __name__ == "__main__":
    main()