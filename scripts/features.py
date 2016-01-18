from sklearn.base import BaseEstimator

import pandas as pd
import numpy as np


class FeatureTransformer(BaseEstimator):
	def __init__(self):
		pass

	def get_feature_names(self):
		"""
		Feature names of the variables in consideration
		"""
		feature_names = []

		feature_names.extend(['accuracy'])

		return np.array(feature_names)

	def fit(self, X, y=None):
		self.fit_transform(X, y)

		return self

	def fit_transform(self, X, y=None):
		# accuracy score for the problem
		# number of problems solved by the user
		
		numeric_features = self.get_features(X)
		
		features = []

		features.append(numeric_features)
		features = np.hstack(features)

		return np.array(features)

	def get_features(self, X):
		accuracy = X.accuracy
		solved_count = X.solved_count_y
		error_count = X.error_count
		attempts = X.attempts

		return np.array([accuracy, solved_count, error_count, attempts]).T

	def transform(self, X):
		numeric_features = self.get_features(X)
		
		features = []

		features.append(numeric_features)
		features = np.hstack(features)

		return np.array(features)
