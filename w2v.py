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

        words = doc.split(' ')
        v = np.zeros(self.Ndim)

        if tfidf:
            indices = np.zeros(len(words), dtype = int)
            error_indices = np.ones_like(indices)
            M = np.zeros((len(words), self.Ndim))
            for i, w in enumerate(words):
                try:
                    indices[i] = self.tfidf.vocabulary_[w]
                except:
                    error_indices[i] = 0
            weights = np.log(self.tfidf.idf_[indices])*error_indices
            for i, w in enumerate(words):
                v += weights[i]*M[i,:]
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
        
    def reduce_dimensionality(self, word_vectors, dimension=2):
        data = np.array(word_vectors)
        pca = dcmp.PCA(n_components=dimension)
        pca.fit(data)
        return pca.transform(data)


class W2vClassifier:

    def __init__(self, w2v):
        self.w2v = w2v
        
    def onehot(self,y):
        output = np.zeros((len(y), 9))
        for i,n in enumerate(y):
            output[i,n-1]=1
        return output

    def vectorize_documents(self, docs, tfidf = True):
        ''' Return a matrix of document vectors '''
        N = len(docs)
        X = np.zeros((N, self.w2v.Ndim))
        for i,doc in enumerate(docs):
            X[i,:] = self.w2v.doc2vec(doc, tfidf)
            if i%500 == 0: print(i)
        return X

    def fit(self, docs, y, tfidf = True):
        X = self.vectorize_documents(docs, tfidf)
        self.X_train = X
        self.clf = NearestCentroid()
        self.clf.fit(X, y)
        self.centroids = self.clf.centroids_ # centroid vectors

    def predict(self, X):
        p = np.zeros((X.shape[0], 9))
        ctr = self.centroids
        
        for n in range(X.shape[0]):
            d = np.zeros(9)
            for i in range(9):
                d[i] = np.linalg.norm(X[n,:] - ctr[i,:])
                d = d/np.linalg.norm(d)
                p[n,:] = d
        return p

    def logloss(self,y_train, X):
        y = self.onehot(y_train)
        N = X.shape[0]
        x = 0
        for i in range(N):
            x+= np.dot(y[i,:], np.log2(X[i,:]))
            x = (-1./N)*x
        return x
        
