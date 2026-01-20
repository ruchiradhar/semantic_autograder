import os
import sys
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
	artifacts_dir: str = os.path.join("artifacts", "data_transformation")
	preprocessor_path: str = os.path.join(artifacts_dir, "preprocessor.pkl")


class DataTransformation:
	def __init__(self, config: "DataTransformationConfig" = None) -> None:
		self.config = config or DataTransformationConfig()

	def get_preprocessor(self) -> ColumnTransformer:
		categorical_cols = [
			"gender",
			"race_ethnicity",
			"parental_level_of_education",
			"lunch",
			"test_preparation_course",
		]
		numeric_cols = [
			"reading_score",
			"writing_score",
		]

		numeric_pipeline = Pipeline(
			steps=[
				("imputer", SimpleImputer(strategy="median")),
				("scaler", StandardScaler()),
			]
		)

		categorical_pipeline = Pipeline(
			steps=[
				("imputer", SimpleImputer(strategy="most_frequent")),
				("onehot", OneHotEncoder(handle_unknown="ignore")),
				("scaler", StandardScaler(with_mean=False)),
			]
		)

		preprocessor = ColumnTransformer(
			transformers=[
				("num", numeric_pipeline, numeric_cols),
				("cat", categorical_pipeline, categorical_cols),
			]
		)
		return preprocessor

	def initiate_data_transformation(
		self, train_path: str, test_path: str, target_col: str = "math_score"
	) -> Tuple[np.ndarray, np.ndarray, str]:
		try:
			os.makedirs(self.config.artifacts_dir, exist_ok=True)
			train_df = pd.read_csv(train_path)
			test_df = pd.read_csv(test_path)
			logging.info("Loaded train/test datasets for transformation")

			X_train = train_df.drop(columns=[target_col])
			y_train = train_df[target_col]
			X_test = test_df.drop(columns=[target_col])
			y_test = test_df[target_col]

			preprocessor = self.get_preprocessor()
			X_train_transformed = preprocessor.fit_transform(X_train)
			X_test_transformed = preprocessor.transform(X_test)

			save_object(self.config.preprocessor_path, preprocessor)
			logging.info("Saved preprocessor to %s", self.config.preprocessor_path)

			train_arr = np.c_[X_train_transformed.toarray() if hasattr(X_train_transformed, 'toarray') else X_train_transformed, y_train.values]
			test_arr = np.c_[X_test_transformed.toarray() if hasattr(X_test_transformed, 'toarray') else X_test_transformed, y_test.values]
			return train_arr, test_arr, self.config.preprocessor_path
		except Exception as e:
			raise CustomException(e, sys)

