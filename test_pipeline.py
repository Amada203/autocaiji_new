import logging
import sys
import traceback

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

def main():
    try:
        logging.info("1. 尝试导入DataPipeline")
        from src.data.data_pipeline import DataPipeline
        logging.info("2. 导入成功，准备初始化")
        
        pipeline = DataPipeline(test_mode=True)
        logging.info("3. 初始化成功，准备运行")
        
        pipeline.run()
        logging.info("4. 运行完成")
        
    except ImportError as e:
        logging.error(f"导入错误: {str(e)}")
        logging.error(traceback.format_exc())
    except Exception as e:
        logging.error(f"运行时错误: {str(e)}")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()