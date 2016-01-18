from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from features import FeatureTransformer

def build_logistic_regression_model():
	ft = FeatureTransformer()
	scaler = StandardScaler()
	clf = LogisticRegression(C=5)


	pipeline = Pipeline([('ft', ft), ('scaler', scaler), ('clf', clf)])
	return pipeline


def build_random_forest_classifier():
	ft = FeatureTransformer()
	clf = RandomForestClassifier()

	pipeline = Pipeline([('ft', ft), ('clf', clf)])

	return pipeline