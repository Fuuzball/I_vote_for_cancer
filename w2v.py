# w2v stuff
# some of this follows
# https://github.com/aubry74/visual-word2vec/blob/master/visual-word2vec.py

import gensim
import os, math
import numpy as np
import pandas as pd
import seaborn as sns
import helpers
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn import svm
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier


class W2vVectorizer:

    def __init__(self, Ndim):
        self.Ndim = Ndim

    def build_vocabulary(self, text):
        sentences = []
        for document in text:
            sts = document.split('.')
            for s in sts:
                words = s.lower().split(' ') # lowercase all the words
                sentences.append(s.split(' '))
            self.sentences = sentences
        return sentences

    def fit(self, text):
        sentences = self.build_vocabulary(text.unique())
        self.tfidf = TfidfVectorizer(
            min_df=1,strip_accents='unicode',lowercase =True,
            analyzer='word', use_idf=True,
            smooth_idf=True, sublinear_tf=True).fit(text.unique())
        self.model = gensim.models.Word2Vec(sentences, size=self.Ndim)
        
    def word2vec(self, word):    
        w = word.lower().rstrip('.')
        return self.model[w]

    def word2weight(self, w, tfidf):
        if tfidf:
            return self.tfidf.idf_[self.tfidf.vocabulary_[w]]
        else:
            return 1

    def doc2vec(self, doc, tfidf=True):
        ### convert a document to a vector

        words = doc.lower().split(' ')
        v = np.zeros(self.Ndim)

        if tfidf:
            indices = np.zeros(len(words), dtype = int)
            error_indices = np.ones_like(indices)
            M = np.zeros((len(words), self.Ndim))
            N = 0
            for i, w in enumerate(words):
                try:
                    indices[i] = self.tfidf.vocabulary_[w]
                    M[i,:] = self.word2vec(w)
                    N += 1
                except:
                    error_indices[i] = 0
            #weights = np.log(self.tfidf.idf_[indices])*error_indices
            #weights = (1-self.tfidf.idf_[indices])*error_indices
            weights = (self.tfidf.idf_[indices])*error_indices
            for i, w in enumerate(words):
                v += weights[i]*M[i,:]
            v = (1./N)*v
        else:
            N = 0
            for w in words:
                try:
                    v += self.word2vec(w)
                    N += 1
                except KeyError:
                    pass # word not in dictionary
            v = (1./N)*v
        return v # return average word vector over the document

    def vectorize_documents(self, docs, tfidf = True):
        ''' Return a matrix of document vectors '''
        N = len(docs)
        X = np.zeros((N, self.Ndim))
        for i,doc in enumerate(docs):
            X[i,:] = self.doc2vec(doc, tfidf)
            if i%500 == 0: print(i)
        return X


def onehot(y):
    output = np.zeros((len(y), 9))
    for i,n in enumerate(y):
        output[i,n-1]=1
    return output

def find_pcs(X, dimension):
    """ compute principle components """
    pca = dcmp.PCA(n_components = dimension)
    pca.fit(X)
    return pca

def train_svm(X, y, C = 1.0, kernel = 'linear', class_weight = 'balanced'):
    '''
    y is a one-hotarray of training labels. Returns the classifier.
    '''
    clf = OneVsRestClassifier(svm.SVC(C=C, kernel=kernel, probability=True,
                                      random_state=0, class_weight=class_weight))

    clf.fit(X, y)
    return clf
