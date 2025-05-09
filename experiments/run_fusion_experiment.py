"""
运行Prophet+LightGBM融合模型实验
"""
import argparse
import os
import sys
import logging

# 确保可以导入项目模块
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from experiments.templates.fusion_model_experiment import FusionModelExperiment

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='运行Prophet+LightGBM融合模型实验')
    parser.add_argument('--config', type=str, 
                        default='experiments/configs/fusion_model_experiment.json',
                        help='配置文件路径')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录，不指定则使用配置文件中的设置')
    
    args = parser.parse_args()
    
    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        logger.error(f"配置文件 {args.config} 不存在")
        return 1
    
    # 初始化实验
    experiment = FusionModelExperiment(args.config)
    
    # 如果指定了输出目录，则覆盖配置
    if args.output_dir:
        experiment.output_dir = args.output_dir
        os.makedirs(args.output_dir, exist_ok=True)
    
    # 运行实验
    try:
        results = experiment.run()
        logger.info(f"实验成功完成")
        return 0
    except Exception as e:
        logger.exception(f"实验运行出错: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 