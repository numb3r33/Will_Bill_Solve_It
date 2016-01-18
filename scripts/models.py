from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from features import FeatureTransformer

def build_logistic_regression_model():
	ft = FeatureTransformer()
	clf = LogisticRegression()


	pipeline = Pipeline([('ft', ft), ('clf', clf)])
	return pipeline