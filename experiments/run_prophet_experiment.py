"""
运行Prophet模型实验
"""
import argparse
from experiments.templates.prophet_experiment import ProphetExperiment

def main():
    parser = argparse.ArgumentParser(description='运行Prophet模型实验')
    parser.add_argument('--config', type=str, default='experiments/configs/prophet_default.json',
                       help='配置文件路径')
    parser.add_argument('--name', type=str, default=None,
                       help='实验名称')
    
    args = parser.parse_args()
    
    # 运行实验
    experiment = ProphetExperiment(args.config, args.name)
    experiment.run()

if __name__ == '__main__':
    main() 