# helper functions for dealing with text

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def get_unique_text(variants_df, text_df, cls, save = None, suppress_output = True):
    """
    Takes pandas DataFrames for variants and text files. The
    class number should be an int corresponding to the tumor
    class. Returns a string with all of the unique text
    corresponding to cls.

    suppress_output suppresses printing additional information
    """

    matching_ids_df = variants_df.loc[variants_df['Class'] == cls]['ID'] # all IDs matching cls
    class_text_df = text_df.loc[text_df['ID'].isin(matching_ids_df)]
    # class_text_df is all the text pertaining to cls (including duplicates)
    unique_text = class_text_df['Text'].unique()
    unique_text_string = '\n'.join(unique_text)

    if not suppress_output:
        cls_size = class_text_df.size
        unique_size = unique_text.shape[0]
        print('Number of entries in class {}: {}'.format(cls, cls_size))
        print('Number of unique texts: {}'.format(unique_size))

    if save is not None:
        print('Saving to file: {}'.format(save))
        f = open(save, 'wt')
        f.write(unique_text_string)
        f.close()

    return unique_text_string

def get_number_instances(docs, vocabulary):
    """
    docs: list of strings
    vocabulary: list of strings

    Returns a numpy array which vectorizes the
    documents according to the number of instances
    of terms in the vocabulary.
    Each -row- of the returned array corresponds
    to a document vector. e.g. to get the
    document vector for document 3, use
    x[3,:]
    """

    cv = CountVectorizer(vocabulary=vocabulary)
    x = cv.fit_transform(docs).toarray()
    return x

def train_test_split(train_variation_file, train_text_file):
    """
    This function takes the variants and text file and splits them into 80% for training and 20% 
    for testing in a stratifiled way according to classes, and returns four DataFrames:
    text_train, text_test, variants_train, variants_test
    """
    from sklearn.model_selection import train_test_split
    import pandas as pd
    y = pd.read_csv(train_variation_file)
    X = pd.read_csv(train_text_file, sep="\|\|", engine="python", skiprows=1, names=["ID", "Text"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y.Class)
    return X_train, X_test, y_train, y_test
