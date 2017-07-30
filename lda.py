import gensim
import os, math
import numpy as np
import pandas as pd
import seaborn as sns
import helpers
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import log_loss

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from gensim import corpora, models
tokenizer = RegexpTokenizer(r'\w+')
en_stop = get_stop_words('en')

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def get_tokens(doc):
    raw = doc.lower()
    tokens = tokenizer.tokenize(raw)
    stopped_tokens = [i for i in tokens if i not in en_stop]

    return stopped_tokens


text_train, var_train = helpers.get_training('./data/training_variants', './data/training_text')
text_test, var_test = helpers.get_test('./data/training_variants', './data/training_text')

print(var_test.head())

T_train = text_train['Text']
#y_train = onehot(var_train['Class'])

T_test = text_test['Text']
#y_test = onehot(var_test['Class'])


len_train = len(T_train)
len_test = len(T_test)

train_tokens = []
for idx, doc in enumerate(T_train):
    if idx % 500 == 0:
        print(idx)
    train_tokens.append(get_tokens(doc))

n_topics = 100
dictionary = corpora.Dictionary(train_tokens)
corpus = [dictionary.doc2bow(text) for text in train_tokens]

print('training lda model...')
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=n_topics, id2word=dictionary, passes=10)
for line in (ldamodel.print_topics(num_topics=n_topics, num_words=5)):
    print(line)

X_train = np.zeros((len_train, n_topics))
for i in range(len_train):
    if i % 500 == 0:
        print(i)
    bow = corpus[i]
    for idx, prob in ldamodel[bow]:
        X_train[i, idx] = prob

X_test = np.zeros((len_test, n_topics))
for i in range(len_test):
    if i % 500 == 0:
        print(i)
    doc = get_tokens(T_test[i])
    bow = dictionary.doc2bow(doc)
    for idx, prob in ldamodel[bow]:
        X_test[i, idx] = prob

for line in (ldamodel.print_topics(num_topics=n_topics, num_words=5)):
    print(line)

np.save('./2nd_layer_data/lda_X_train', X_train)
np.save('./2nd_layer_data/lda_X_test', X_test) 
