from sklearn.linear_model import LogisticRegression, SGDClassifier, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier

import xgboost as xgb

from features import FeatureTransformer

def build_logistic_regression_model(X, X_test):
	ft = FeatureTransformer(X, X_test)
	scaler = MinMaxScaler()
	clf = LogisticRegression(C=1.)


	pipeline = Pipeline([('ft', ft), ('scaler', scaler), ('clf', clf)])
	return pipeline

def build_elastic_net_model(X, X_test):
	ft = FeatureTransformer(X, X_test)
	scaler = MinMaxScaler()
	clf = ElasticNet()


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
	scaler = MinMaxScaler()
	clf = KNeighborsClassifier(n_neighbors=7, weights='distance')

	pipeline = Pipeline([('ft', ft), ('scaler', scaler), ('clf', clf)])

	return pipeline


def build_sgd_classifier(X, X_test):
	ft = FeatureTransformer(X, X_test)
	scaler = MinMaxScaler()
	clf = SGDClassifier(penalty='elasticnet')

	pipeline = Pipeline([('ft', ft), ('scaler', scaler), ('clf', clf)])

	return pipeline

def build_extreme_gradient_boosting(X, X_test):
	ft = FeatureTransformer(X, X_test)
	clf = xgb.XGBClassifier(n_estimators=500, max_depth=4)

	pipeline = Pipeline([('ft', ft), ('clf', clf)])

	return pipeline
