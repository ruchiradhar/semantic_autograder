import argparse
import os
import sys
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


@dataclass
class StudentData:
	gender: str
	race_ethnicity: str
	parental_level_of_education: str
	lunch: str
	test_preparation_course: str
	reading_score: float
	writing_score: float

	def to_df(self) -> pd.DataFrame:
		return pd.DataFrame([
			{
				"gender": self.gender,
				"race_ethnicity": self.race_ethnicity,
				"parental_level_of_education": self.parental_level_of_education,
				"lunch": self.lunch,
				"test_preparation_course": self.test_preparation_course,
				"reading_score": self.reading_score,
				"writing_score": self.writing_score,
			}
		])


class PredictPipeline:
	def __init__(self,
				 model_path: str = None,
				 preprocessor_path: str = None) -> None:
		self.model_path = model_path or os.path.join("artifacts", "model_trainer", "model.pkl")
		self.preprocessor_path = preprocessor_path or os.path.join(
			"artifacts", "data_transformation", "preprocessor.pkl"
		)

	def predict(self, df: pd.DataFrame) -> np.ndarray:
		try:
			preprocessor = load_object(self.preprocessor_path)
			model = load_object(self.model_path)
			X = preprocessor.transform(df)
			preds = model.predict(X)
			return np.asarray(preds)
		except Exception as e:
			raise CustomException(e, sys)


def cli(argv: List[str] = None) -> None:
	parser = argparse.ArgumentParser(description="Predict math score from student attributes")
	parser.add_argument("--gender", default="female")
	parser.add_argument("--race", dest="race_ethnicity", default="group B")
	parser.add_argument("--parental", dest="parental_level_of_education", default="bachelor's degree")
	parser.add_argument("--lunch", default="standard")
	parser.add_argument("--prep", dest="test_preparation_course", default="none")
	parser.add_argument("--reading", dest="reading_score", type=float, default=72)
	parser.add_argument("--writing", dest="writing_score", type=float, default=74)
	args = parser.parse_args(argv)

	data = StudentData(
		gender=args.gender,
		race_ethnicity=args.race_ethnicity,
		parental_level_of_education=args.parental_level_of_education,
		lunch=args.lunch,
		test_preparation_course=args.test_preparation_course,
		reading_score=args.reading_score,
		writing_score=args.writing_score,
	)
	df = data.to_df()
	preds = PredictPipeline().predict(df)
	print(float(preds[0]))


if __name__ == "__main__":
	cli()

