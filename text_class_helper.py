import time
from datetime import timedelta
import random

import numpy as np
import pandas as pd
import helpers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize


class TextClfHelper(object):

    def __init__(self, fname_var=None, fname_txt=None, test_frac=0.2,  seed=0):

        random.seed(seed)

        # Set default path to data
        if fname_var is None:
            fname_var = './data/training_variants'
        if fname_txt is None:
            fname_txt = './data/training_text'

        # Import data
        print('Importing csv...')
        self.df_var = pd.read_csv(fname_var)
        self.df_txt = pd.read_csv(fname_txt, sep='\|\|', engine='python', skiprows=1, names=['ID', 'Text'])
        self.df_full = self.df_var.merge(self.df_txt, how='inner', on='ID')

        # Get train / test split
        self.df_train, self.df_test = train_test_split(self.df_full, test_size=test_frac)
        self.txt_train = self.df_train['Text'].values.tolist()
        self.cls_train = self.df_train['Class'].values
        self.txt_test = self.df_test['Text'].values.tolist()
        self.cls_test = self.df_test['Class'].values

        self.one_hot_train = label_binarize(self.cls_train, classes=range(1, 10))
        self.one_hot_test = label_binarize(self.cls_test, classes=range(1, 10))

        

if __name__ == '__main__':
    pipeline = TextClfHelper()
    print(pipeline.txt_test[0])

