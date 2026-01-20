import os
import sys
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
	artifacts_dir: str = os.path.join("artifacts", "model_trainer")
	model_path: str = os.path.join(artifacts_dir, "model.pkl")


class ModelTrainer:
	def __init__(self, config: "ModelTrainerConfig" = None) -> None:
		self.config = config or ModelTrainerConfig()

	def initiate_model_training(self, train_arr: np.ndarray, test_arr: np.ndarray) -> Tuple[str, float]:
		try:
			os.makedirs(self.config.artifacts_dir, exist_ok=True)

			X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
			X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

			models: Dict[str, object] = {
				"LinearRegression": LinearRegression(),
				"DecisionTree": DecisionTreeRegressor(random_state=42),
				"RandomForest": RandomForestRegressor(random_state=42),
				"GradientBoosting": GradientBoostingRegressor(random_state=42),
			}

			param_grid: Dict[str, Dict[str, object]] = {
				"RandomForest": {"n_estimators": [100, 200], "max_depth": [None, 10, 20]},
				"GradientBoosting": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
				"DecisionTree": {"max_depth": [None, 5, 10]},
			}

			best_name, best_model, scores = evaluate_models(
				X_train, y_train, X_test, y_test, models, param_grid
			)

			save_object(self.config.model_path, best_model)
			logging.info("Saved best model '%s' to %s", best_name, self.config.model_path)

			y_pred = best_model.predict(X_test)
			r2 = r2_score(y_test, y_pred)
			logging.info("Best model R2 on test: %.4f", r2)
			return best_name, float(r2)
		except Exception as e:
			raise CustomException(e, sys)

