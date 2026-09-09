import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            logging.info("Starting complete training pipeline")
            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

            data_transformation = DataTransformation()
            train_arr, test_arr = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

            model_trainer = ModelTrainer()
            r2_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
            logging.info(f"Training pipeline completed successfully with R2 Score: {r2_score}")
            return r2_score
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = TrainPipeline()
    score = pipeline.run_pipeline()
    print(f"Training completed successfully. Model R2 Score: {score}")
