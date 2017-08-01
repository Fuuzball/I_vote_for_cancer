# train the w2v model and save the results

import gensim
import os, math
import numpy as np
import pandas as pd
import seaborn as sns
import helpers
import matplotlib.pyplot as plt
from w2v import *

LOAD_FROM_FILE = True # set to False to retrain w2v

X_text, y = helpers.get_training('./data/training_variants', './data/training_text')
X_text = X_text.Text
y_train = y.Class.values
ids = y.ID

if LOAD_FROM_FILE:
    X_train = np.load('w2v_features.npy')

else:
    print('training the w2v vectorizer')
    w2v = W2vVectorizer(100)
    w2v.fit(X_text)
    print('Vectorizing the training data')
    X_train = w2v.vectorize_documents(X_text)

    # uncomment this to save the features
    #np.save('w2v_features', X_train)
    # X_train is a matrix of features.
    
clf = train_svm(X_train, y_train)

y_train_prob = clf.predict_proba(X_train)
output = np.insert(y_train_prob, 0, ids,axis = 1)
helpers.submission('w2v_probabilities', output)

