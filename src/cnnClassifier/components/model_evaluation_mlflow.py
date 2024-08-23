from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
import tensorflow as tf
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json
from cnnClassifier import logger
import os

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _valid_generator(self):

        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )


    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    
    def log_into_mlflow(self):
        logger.info("In log_into" + self.config.mlflow_uri)
        logger.info("trackin set " + str(mlflow.is_tracking_uri_set()))
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        mlflow.set_registry_uri(self.config.mlflow_uri)
        logger.info("now trackin set " + str(mlflow.is_tracking_uri_set()))
        logger.info("registry uri " + mlflow.get_registry_uri())
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        logger.info("tracking" + tracking_url_type_store)
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )
            # Model registry does not work with file store
            if tracking_url_type_store != "file":
                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                logger.info("If Saving") 
                mlflow.sklearn.log_model(
                    sk_model=self.model,
                    artifact_path="sklearn-model",
                    registered_model_name="sk-learn-random-forest-reg-model",)
            else:
                logger.info("Else Saving") 
                mlflow.sklearn.log_model(
                    sk_model=self.model,
                    artifact_path="sklearn-model")
            logger.info("Saved Succefully")    
                