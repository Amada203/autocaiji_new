import pandas as pd
import joblib
import logging
from src.models.fusion_model import PriceChangePredictor

logger = logging.getLogger("model_predictor")

class ModelPredictor:
    def __init__(self, model_path: str, features_path: str = None):
        try:
            self.model = joblib.load(model_path)
            logger.info(f"模型已加载: {model_path}")
            # 加载特征名列表，优先从外部文件加载，便于后续拓展
            self.features = None
            model_features = list(self.model.feature_name_) if hasattr(self.model, 'feature_name_') else None
            logger.info(f"模型自带特征名: {model_features}")
            if features_path is not None:
                import json
                with open(features_path, 'r') as f:
                    self.features = json.load(f)
                    logger.info(f"加载特征名列表: {self.features}")
                # 检查一致性
                if model_features is not None and self.features != model_features:
                    logger.error(f"警告：加载的特征名与模型自带特征名不一致！\nfeatures_path: {self.features}\nmodel.feature_name_: {model_features}")
                    raise ValueError("加载的特征名与模型自带特征名不一致，请检查 features.json 和模型训练时用的特征！")
            elif model_features is not None:
                self.features = model_features
                logger.info(f"从模型属性加载特征名: {self.features}")
            else:
                logger.warning("未能加载特征名列表，推理时将使用全部特征（不推荐）")
                raise ValueError("无法获取特征名列表，模型推理无法保证特征顺序一致！")
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        输入特征，输出预测概率
        """
        try:
            logger.info(f"推理输入特征: {list(features.columns)}")
            # 只保留训练时用的特征，顺序严格一致
            if self.features is not None:
                # 检查输入特征是否包含全部所需特征
                missing = [f for f in self.features if f not in features.columns]
                if missing:
                    logger.error(f"推理输入缺少特征: {missing}")
                    raise ValueError(f"推理输入缺少特征: {missing}")
                features = features[self.features]
            proba = self.model.predict_proba(features)[:, 1]
            result = features.copy()
            result['predict_proba'] = proba
            return result
        except Exception as e:
            logger.error(f"模型推理失败: {str(e)}")
            raise 