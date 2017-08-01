# train the w2v model and save the results

import gensim
import os, math
import numpy as np
import pandas as pd
import seaborn as sns
import helpers
import matplotlib.pyplot as plt
from w2v import *

LOAD_FROM_FILE = False # set to False to retrain w2v

# load all labeled training data
X1, y1 = helpers.get_training('./data/training_variants', './data/training_text')
X2, y2 = helpers.get_test('./data/training_variants', './data/training_text')
X_text = pd.concat([X1, X2])
y = pd.concat([y1, y2])
X_text = X_text.Text
y_train = y.Class.values
ids = y.ID

# load the unlabeled test data as well
test_data = pd.read_csv("./data/test_text", sep="\|\|", engine="python", skiprows=1, names=["ID", "Text"])
test_text = test_data.Text
test_ids = test_data.ID

if LOAD_FROM_FILE:
    X_train = np.load('w2v_train_features.npy')
    X_test = np.load('w2v_test_features.npy')
else:
    print('training the w2v vectorizer')
    w2v = W2vVectorizer(100)
    w2v.fit(X_text)
    print('Vectorizing the training data')
    X_train = w2v.vectorize_documents(X_text)
    X_test = w2v.vectorize_documents(test_text)
    # uncomment this to save the features
    np.save('w2v_train_features', X_train)
    np.save('w2v_test_features', X_test)
    # X_train is a matrix of features.
    
clf = train_svm(X_train, y_train)

y_train_prob = clf.predict_proba(X_train)
y_test_prob = clf.predict_proba(X_test)
output = np.insert(y_train_prob, 0, ids,axis = 1)
helpers.submission('w2v_train_probabilities', output)

test_output = np.insert(y_test_prob, 0, test_ids,axis = 1)
helpers.submission('w2v_test_probabilities', test_output)





