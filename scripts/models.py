from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier

import xgboost as xgb

from features import FeatureTransformer

def build_logistic_regression_model(X, X_test):
	ft = FeatureTransformer(X, X_test)
	scaler = StandardScaler()
	clf = LogisticRegression(C=1.)


	pipeline = Pipeline([('ft', ft), ('scaler', scaler), ('clf', clf)])
	return pipeline


def build_random_forest_classifier(X, X_test):
	ft = FeatureTransformer(X, X_test)
	clf = RandomForestClassifier(n_estimators=500, criterion='gini', n_jobs=-1)

	pipeline = Pipeline([('ft', ft), ('clf', clf)])

	return pipeline

def build_extra_trees_classifier(X, X_test):
	ft = FeatureTransformer(X, X_test)
	clf = ExtraTreesClassifier(n_estimators=100)

	pipeline = Pipeline([('ft', ft), ('clf', clf)])

	return pipeline

def build_knn_classifier(X, X_test):
	ft = FeatureTransformer(X, X_test)
	clf = KNeighborsClassifier(n_neighbors=5, weights='distance')

	pipeline = Pipeline([('ft', ft), ('clf', clf)])

	return pipeline


def build_sgd_classifier(X, X_test):
	ft = FeatureTransformer(X, X_test)
	scaler = StandardScaler()
	clf = SGDClassifier(loss='hinge', penalty='l2')

	pipeline = Pipeline([('ft', ft), ('scaler', scaler), ('clf', clf)])

	return Pipeline

def build_extreme_gradient_boosting(X, X_test):
	ft = FeatureTransformer(X, X_test)
	clf = xgb.XGBClassifier(n_estimators=500)

	pipeline = Pipeline([('ft', ft), ('clf', clf)])

	return pipeline
