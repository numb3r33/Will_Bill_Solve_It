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

		feature_names.extend(['accuracy', 'per_people_solved',
			                  'num_problems_solved', 'num_problems_solved_incorrectly',
			                  'user_id', 'problem_id'])
		feature_names.extend(self.categorical_features_columns)
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

		features = []

		features.append(numeric_features)
		features.append(categorical_features)
		features = np.hstack(features)

		return np.array(features)

	def get_features(self, X):
		accuracy = X.accuracy # accuracy score for the problem
		per_people_solved = X.solved_count_y * 1. / (X.solved_count_y + X.error_count) # percentage of people who solved it correctly
		num_problems_solved = X.solved_count_x
		num_problems_solved_incorrectly = X.attempts
		user_id = X.user_id
		problem_id = X.problem_id
		
		return np.array([accuracy, per_people_solved, 
			             num_problems_solved, num_problems_solved_incorrectly,
			             user_id, problem_id
			            ]).T

	def get_skills(self, X):
		"""
		Return features regarding skill set of the user
		"""

		self.skill_features = ['Befunge', 'C', 'C#', 'C++', 'C++ (g++ 4.8.1)',
                          'Clojure', 'Go', 'Haskell', 'Java', 'Java (openjdk 1.7.0_09)',
                          'JavaScript', 'JavaScript(Node.js)', 'JavaScript(Rhino)', 'Lisp',
                          'Objective-C', 'PHP', 'Pascal', 'Perl', 'Python', 'Python 3',
                          'R(RScript)', 'Ruby', 'Rust', 'Scala', 'Text', 'Whenever']

		return np.array(X[self.skill_features]).T

	def get_categorical_features(self, X):
		self.categorical_features_columns = ['level', 'user_type', 'tag1']
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
		
		features = []

		features.append(numeric_features)
		features.append(categorical_features)

		features = np.hstack(features)

		return np.array(features)
