from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_trainer import Prediction
from cnnClassifier import logger

STAGE_NAME = "Training"

class ModelPredictionPipeline:
    def __init__(self, img):
        self.img = img

    def main(self):
        config = ConfigurationManager()
        prediction_config = config.get_prediction_config()
        prediction = Prediction(config=prediction_config)
        prediction.predict(self.img)

if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelPredictionPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e