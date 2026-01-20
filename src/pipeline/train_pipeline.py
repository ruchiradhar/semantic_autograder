import os
import sys

from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_training import ModelTrainer, ModelTrainerConfig
from src.exception import CustomException
from src.logger import logging


def main() -> None:
	try:
		project_root = os.getcwd()
		source_csv = os.path.join(project_root, "notebooks", "data", "stud.csv")

		ingestion = DataIngestion(DataIngestionConfig())
		train_path, test_path = ingestion.initiate_data_ingestion(source_csv)

		transform = DataTransformation(DataTransformationConfig())
		train_arr, test_arr, preproc_path = transform.initiate_data_transformation(
			train_path, test_path, target_col="math_score"
		)
		logging.info("Preprocessor saved to %s", preproc_path)

		trainer = ModelTrainer(ModelTrainerConfig())
		best_name, r2 = trainer.initiate_model_training(train_arr, test_arr)
		logging.info("Training complete. Best model: %s (R2=%.4f)", best_name, r2)
		print(f"Best model: {best_name} | R2: {r2:.4f}")
	except Exception as e:
		raise CustomException(e, sys)


if __name__ == "__main__":
	main()

