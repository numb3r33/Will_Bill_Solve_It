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
		accuracy_as_feature = self.get_accuracy(X)

		features = []

		features.append(accuracy_as_feature)
		features = np.hstack(features)

		return np.array(features)

	def get_accuracy(self, X):
		"""
		Returns a accuracy associated with a problem
		"""
		accuracy = X.accuracy

		return accuracy.reshape(-1, 1)

	def transform(self, X):
		accuracy_as_feature = self.get_accuracy(X)

		features = []

		features.append(accuracy_as_feature)
		features = np.hstack(features)
		
		return np.array(features)
		