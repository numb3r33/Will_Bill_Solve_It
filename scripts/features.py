from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np


class FeatureTransformer(BaseEstimator):
	def __init__(self, X, X_test):
		self.X = X
		self.X_test = X_test

	def get_feature_names(self):
		"""
		Feature names of the variables in consideration
		"""
		feature_names = []

		feature_names.extend(['accuracy', 'solved_count_y', 'attempts',
			                  'user_capability_ratio', 'solved_count_x',
			                  'error_count', 'problem_difficulty_ratio',
			                  'user_id', 'problem_id'])
		
		feature_names.extend(self.skill_features)


		return np.array(feature_names)

	def fit(self, X, y=None):
		self.fit_transform(X, y)

		return self

	def fit_transform(self, X, y=None):
		# accuracy score for the problem
		# number of problems solved by the user
		
		numeric_features = self.get_features(X)
		categorical_features = self.get_categorical_features(X)
		skill_features = self.get_skills(X)
		problem_types = self.get_problem_types(X)

		features = []

		features.append(numeric_features)
		features.append(categorical_features)
		features.append(skill_features)
		features.append(problem_types)

		features = np.hstack(features)

		return np.array(features)

	def get_features(self, X):
		# accuracy score for the problem
		accuracy = X.accuracy

		num_problems_solved = X.solved_count_y
		num_incorrect_submissions = X.attempts

		user_capability_ratio = num_problems_solved / (num_problems_solved + num_incorrect_submissions) * 1.

		num_times_solved_correctly = X.solved_count_x
		num_times_solved_incorrectly = X.error_count

		problem_difficulty_ratio = num_times_solved_correctly / (num_times_solved_correctly + num_times_solved_incorrectly) * 1.
		
		user_id = X.user_id
		problem_id = X.problem_id
		
		return np.array([accuracy, num_problems_solved,
						 num_times_solved_correctly,
						 num_times_solved_incorrectly,
						 user_capability_ratio,
						 problem_difficulty_ratio,
						 user_id,
						 problem_id]).T

	def get_skills(self, X):
		"""
		Return features regarding skill set of the user
		"""

		self.skill_features = X.columns[17:]

		return np.array(X[self.skill_features])

	def get_problem_types(self, X):
		"""
		Return features regarding problem type
		"""

		self.problem_types = X.columns[43:]

		return np.array(X[self.problem_types])


	def get_categorical_features(self, X):
		self.categorical_features_columns = ['level']
		categorical_features = []

		for cat in self.categorical_features_columns:
			lbl = LabelEncoder()

			lbl.fit(pd.concat([self.X[cat], self.X_test[cat]], axis=0))

			categorical_features.append(lbl.transform(X[cat]))

		return np.array(categorical_features).T


	def transform(self, X):
		numeric_features = self.get_features(X)
		categorical_features = self.get_categorical_features(X)
		skill_features = self.get_skills(X)
		problem_types = self.get_problem_types(X)
		
		features = []

		features.append(numeric_features)
		features.append(categorical_features)
		features.append(skill_features)
		features.append(problem_types)

		features = np.hstack(features)

		return np.array(features)
