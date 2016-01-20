from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb

from features import FeatureTransformer

def build_logistic_regression_model(X, X_test):
	ft = FeatureTransformer(X, X_test)
	scaler = StandardScaler()
	clf = LogisticRegression(C=5)


	pipeline = Pipeline([('ft', ft), ('scaler', scaler), ('clf', clf)])
	return pipeline


def build_random_forest_classifier(X, X_test):
	ft = FeatureTransformer(X, X_test)
	clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)

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
	clf = xgb.XGBClassifier()

	pipeline = Pipeline([('ft', ft), ('clf', clf)])

	return pipeline
