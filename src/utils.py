import os
import sys
import pickle
from typing import Any, Dict, Tuple, Optional

import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging


def ensure_dir(path: str) -> None:
	os.makedirs(os.path.dirname(path), exist_ok=True)


def save_object(file_path: str, obj: Any) -> None:
	try:
		ensure_dir(file_path)
		with open(file_path, "wb") as f:
			pickle.dump(obj, f)
		logging.info("Saved object to %s", file_path)
	except Exception as e:
		raise CustomException(e, sys)  # type: ignore[name-defined]


def load_object(file_path: str) -> Any:
	try:
		with open(file_path, "rb") as f:
			obj = pickle.load(f)
		logging.info("Loaded object from %s", file_path)
		return obj
	except Exception as e:
		raise CustomException(e, sys)  # type: ignore[name-defined]


def evaluate_models(
	X_train: np.ndarray,
	y_train: np.ndarray,
	X_test: np.ndarray,
	y_test: np.ndarray,
	models: Dict[str, Any],
	param_grid: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[str, Any, Dict[str, float]]:
	"""Train and evaluate multiple models returning best by R2.

	Returns (best_name, best_model, scores)
	"""
	scores: Dict[str, float] = {}
	best_name = None
	best_model = None
	best_score = float("-inf")

	for name, model in models.items():
		try:
			if param_grid and name in param_grid and param_grid[name]:
				grid = GridSearchCV(model, param_grid[name], cv=3, n_jobs=-1)
				grid.fit(X_train, y_train)
				candidate = grid.best_estimator_
			else:
				model.fit(X_train, y_train)
				candidate = model

			preds = candidate.predict(X_test)
			r2 = r2_score(y_test, preds)
			scores[name] = r2
			logging.info("Model %s achieved R2=%.4f", name, r2)
			if r2 > best_score:
				best_score = r2
				best_name = name
				best_model = candidate
		except Exception as e:
			logging.exception("Failed training model %s", name)
			raise CustomException(e, sys)  # type: ignore[name-defined]

	assert best_name is not None and best_model is not None
	return best_name, best_model, scores

