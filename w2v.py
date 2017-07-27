# w2v stuff
# some of this follows
# https://github.com/aubry74/visual-word2vec/blob/master/visual-word2vec.py

import gensim
import os, math
import numpy as np
import pandas as pd
import seaborn as sns
import helpers
%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

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

    def fit(self, text):
        sentences = self.build_sentences(text)
        self.model = gensim.models.Word2Vec(sentences, size=self.Ndim)
        
    def word2vec(self, word):    
        w = word.lower().rstrip('.')
        return self.model[w]

    def doc2vec(self, doc):
        ### convert a document to a vector

        words = doc.split(' ')
        N = 0
        v = np.zeros(self.Ndim)

        for w in words:
            try:
                v += self.word2vec(w)
                N += 1
            except KeyError:
                pass # word not in dictionary

        v = (1./N)*v
        return v # return average word vector over the document
        

    
    
