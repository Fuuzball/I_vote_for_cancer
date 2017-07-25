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

class w2v_featurizer:

    def build_vocabulary(self, text):
        sentences = []
        for document in text:
            sts = document.split('.')
            for s in sts:
                words = s.lower().split(' ') # lowercase all the words
                sentences.append(s.split(' '))
            self.sentences = sentences 

    
