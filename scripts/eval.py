from sklearn.metrics import accuracy_score
from sklearn.cross_validation import StratifiedShuffleSplit

import numpy as np
from collections import Counter

def eval_models(models, X, y):

	cv = StratifiedShuffleSplit(y, n_iter=5, test_size=0.3)
	scores = []

	for train, test in cv:
		preds_combined = []

		for clf in models:
			X_train, y_train = X.iloc[train], y.iloc[train]
			X_test, y_test = X.iloc[test], y.iloc[test]
			clf.fit(X_train, y_train)
			preds = clf.predict(X_test)

			print("accuracy score: %f" % accuracy_score(y_test, preds))

			preds_combined.append(preds)

		preds_combined = majority_voting(preds_combined)
		scores.append(accuracy_score(y_test, preds_combined))

		print("combined score: %f" % scores[-1])

	return scores


def majority_voting(preds):
    """
    Given an array of predictions from various classifiers
    return single array with ensemble of predictions based on
    simple majority voting
    
    Input: list of list [[y1, y2, y3, ..], [y1, y2, y3, ...], ..] 
    Output: final prediction [y1, y2, y3, ..]
    """
    length = [len(pred) for pred in preds]
    
    if len(set(length)) != 1:
        raise ValueError('Predictions must be of the same length')
    
    pred_matrix = np.matrix(preds)
    ensemble_preds = []
    
    for i in range(len(preds[0])):
        pred_column = np.array(pred_matrix[:, i]).ravel()
        common_pred = Counter(pred_column)
        most_common = common_pred.most_common()[0][0]
        
        ensemble_preds.append(most_common)
    
    return ensemble_preds