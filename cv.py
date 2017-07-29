import numpy as np
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelBinarizer

def kfold_score(clf, X,y, splits=3):
    '''
    clf is the classifier. X is all the training data,
    y is the labels. Returns average log-loss over
    the k folds
    '''
    lb = LabelBinarizer()
    lb.fit(y)
    k_fold = KFold(n_splits=splits)
    values = []
    for train, test in k_fold.split(X):
        clf.fit(X[train], y[train])
        y_test_prob = clf.predict_proba(X[test])
        y_true = lb.transform(y[test])
        values.append(log_loss(y_true, y_test_prob))
    return np.mean(values)
    
