import os
import sys
from dataclasses import dataclass
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
	artifacts_dir: str = os.path.join("artifacts", "data_ingestion")
	raw_data_path: str = os.path.join(artifacts_dir, "raw.csv")
	train_data_path: str = os.path.join(artifacts_dir, "train.csv")
	test_data_path: str = os.path.join(artifacts_dir, "test.csv")


class DataIngestion:
	def __init__(self, config: "DataIngestionConfig" = None) -> None:
		self.config = config or DataIngestionConfig()

	def initiate_data_ingestion(self, source_csv: str) -> Tuple[str, str]:
		"""Read from `source_csv`, split, and persist train/test CSVs.

		Returns (train_path, test_path)
		"""
		logging.info("Starting data ingestion from %s", source_csv)
		try:
			os.makedirs(self.config.artifacts_dir, exist_ok=True)
			df = pd.read_csv(source_csv)
			df.to_csv(self.config.raw_data_path, index=False)
			logging.info("Saved raw data to %s", self.config.raw_data_path)

			train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
			train_df.to_csv(self.config.train_data_path, index=False)
			test_df.to_csv(self.config.test_data_path, index=False)
			logging.info(
				"Split data: train=%d, test=%d", len(train_df), len(test_df)
			)
			return self.config.train_data_path, self.config.test_data_path
		except Exception as e:
			raise CustomException(e, sys)
