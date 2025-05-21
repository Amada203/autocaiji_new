import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_processor import DataProcessor
from src.features.feature_engineering import FeatureEngineering
from src.utils.visualization import DataVisualizer

def main():
    # 加载数据
    data_processor = DataProcessor('data/raw/sku_price_data.csv')
    df = data_processor.preprocess_data()
    
    # 特征工程
    feature_engineering = FeatureEngineering(df)
    df = feature_engineering.build_all_features()
    
    # 创建可视化器
    visualizer = DataVisualizer(df)
    
    # 创建输出目录
    output_dir = 'notebooks/visualization_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存所有图表
    visualizer.save_plots(output_dir)
    
    # 为特定SKU生成详细分析
    # 获取价格变动最频繁的SKU
    top_sku = df.groupby('sku_id')['price_change_flag'].sum().sort_values(ascending=False).index[0]
    
    # 生成该SKU的详细分析图
    fig = visualizer.plot_price_trend(top_sku)
    fig.write_html(os.path.join(output_dir, f'sku_{top_sku}_analysis.html'))
    
    print(f"可视化结果已保存到目录: {output_dir}")
    print(f"请打开以下文件查看分析结果：")
    print(f"1. {output_dir}/price_change_distribution.html - 价格变动分布分析")
    print(f"2. {output_dir}/weekly_pattern.html - 每周价格变动模式")
    print(f"3. {output_dir}/price_change_pattern.html - Top SKU价格变动模式")
    print(f"4. {output_dir}/sku_{top_sku}_analysis.html - SKU {top_sku} 详细分析")

if __name__ == "__main__":
    main() 