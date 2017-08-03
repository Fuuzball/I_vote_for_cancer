import numpy as np
import helpers
import pandas as pd
import seaborn as sns
import helpers
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import decomposition as dcmp
from sklearn.metrics import log_loss
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm

class pca_placeholder: # placeholder class for pca. Does nothing.
    def transform(self, X):
        return X
    def fit(self, X):
        pass

def train_ensemble_classifier(predictions, y, C = 1.):
    X = np.concatenate(predictions, axis=1)
    clf = OneVsRestClassifier(svm.SVC(C=C, kernel='linear', probability=True, random_state=0, class_weight='balanced'))
    print('Training ensemble classifier')
    clf.fit(X, y)
    return clf

def train_svm(X, y):
    clf = OneVsRestClassifier(svm.SVC(C=1., kernel='linear', probability=True, random_state=0, class_weight='balanced'))
    print('Training SVM')
    clf.fit(X, y)
    return clf

def train_count_classifier(X_text, vocab, y_train, pca_dim = None):
    vectorizer=CountVectorizer(vocabulary=vocab.unique(), lowercase=False)
    data = vectorizer.transform(X_text.unique()).toarray() # featurize the text by the vocabulary
    if pca_dim is None:
        pca = pca_placeholder()
    else:
        pca = dcmp.PCA(n_components = pca_dim)
        pca.fit(data)
    X = pca.transform(vectorizer.transform(X_text).toarray()) # (possibly reduced dimension) features
    scale_factor = np.sqrt(np.mean(np.square(X)))
    clf = train_svm(X/scale_factor, y_train)
    return clf, vectorizer, pca, X, scale_factor

def train_w2v_classifier(X_w2v, y_train, pca_dim = None):
    if pca_dim is None:
        pca = pca_placeholder()
    else:
        pca = dcmp.PCA(n_components = pca_dim)
        pca.fit(X_w2v)
    X = pca.transform(X_w2v)
    scale_factor = np.sqrt(np.mean(np.square(X)))
    clf = train_svm(X/scale_factor, y_train)
    return clf, pca, X, scale_factor

def construct_training_data():
    X1, y1 = helpers.get_training('./data/training_variants', './data/training_text')
    X2, y2 = helpers.get_test('./data/training_variants', './data/training_text')
    X_text = pd.concat([X1, X2]).Text
    y = pd.concat([y1, y2])
    y_train = y.Class
    genes = y.Gene
    variants = y.Variation
    return X_text, genes, variants, y_train

def load_train_test_data(train_file, test_file):
    X1 = np.loadtxt(train_file,skiprows=1, delimiter=',')
    X2 = np.loadtxt(test_file,skiprows=1, delimiter=',')
    return np.concatenate((X1,X2))[:,1:]

def load_single(train_file):
    X1 = np.loadtxt(train_file,skiprows=1, delimiter=',')
    return X1[:,1:]


if __name__ == '__main__':
    X_text, genes, variants, y_train = construct_training_data()
    #test_data = pd.read_csv("./data/test_text", sep="\|\|", engine="python", skiprows=1, names=["ID", "Text"])
    #test_text = test_data.Text
    #test_data = pd.read_csv("./data/test_variants") # overwrite test_data as it is not needed
    #test_gene = test_data.Gene
    #test_variant = test_data.Variation
    
    #w2v_train_features = np.load('./data/w2v_train_features.npy')
    #w2v_test_features = np.load('./data/w2v_test_features.npy')

    # --- w2v classifier ---

    #clf_w2v, pca_w2v, X_w2v, k_w2v = train_w2v_classifier( w2v_train_features, y_train)
    #helpers.submission('./2nd_layer_data/w2v/train_prob',clf_w2v.predict_proba(X_w2v/k_w2v))
    #X_w2v_test = pca_w2v.transform(w2v_test_features)
    #helpers.submission('./2nd_layer_data/w2v/true_test_prob',clf_w2v.predict_proba(X_w2v_test/k_w2v))

    # --- variants classifier ---
    #clf, vec, pca, X, k = train_count_classifier(X_text, variants, y_train, pca_dim= 50)
    #helpers.submission('./2nd_layer_data/only_var/train_prob',clf.predict_proba(X/k))
    #X_test = pca.transform(vec.transform(test_text).toarray())
    #helpers.submission('./2nd_layer_data/only_var/true_test_prob',clf.predict_proba(X_test/k))
    
    # --- gene classifier ---
    #clf, vec, pca, X, k = train_count_classifier(X_text, genes, y_train, pca_dim= 50)
    #helpers.submission('./2nd_layer_data/only_gene/train_prob',clf.predict_proba(X/k))
    #X_test = pca.transform(vec.transform(test_text).toarray())
    #helpers.submission('./2nd_layer_data/only_gene/true_test_prob',clf.predict_proba(X_test/k))
    
    #clf_gene, vec_gene, pca_gene, X_gene, k_gene = train_count_classifier(X_text, genes, y_train, pca_dim= 50)
    #print(log_loss(y_train, clf_gene.predict_proba(X_gene/k_gene)))

    
    xgb_train_prob = load_train_test_data('./2nd_layer_data/xgboost/train_prob', './2nd_layer_data/xgboost/test_prob')
    tfidf_train_prob = load_train_test_data('./2nd_layer_data/tfidf/train_prob', './2nd_layer_data/tfidf/test_prob')
    w2v_train_prob = load_single('./2nd_layer_data/w2v/train_prob')
    var_train_prob = load_single('./2nd_layer_data/only_var/train_prob')
    gene_train_prob = load_single('./2nd_layer_data/only_gene/train_prob')

    base_prob = np.zeros_like(xgb_train_prob)
    for i in range(base_prob.shape[0]):
        base_prob[i,:] = np.array([ 0.17103282,  0.13610358,  0.02679916,  0.20656429,  0.07286962, 0.08280638,  0.28696176,  0.00572117,  0.01114122])

    #predictions = [xgb_train_prob, w2v_train_prob, var_train_prob, gene_train_prob]
    predictions = [xgb_train_prob, tfidf_train_prob, w2v_train_prob, var_train_prob, gene_train_prob]
    X_ensemble = np.concatenate(predictions, axis = 1)
    clf_ensemble = train_ensemble_classifier(predictions, y_train)
    print(log_loss(y_train, clf_ensemble.predict_proba(X_ensemble)))
    print('building test submission')

    xgb_test_prob = load_single('./2nd_layer_data/xgboost/true_test_prob')
    tfidf_test_prob = load_single('./2nd_layer_data/tfidf/true_test_prob')
    w2v_test_prob = load_single('./2nd_layer_data/w2v/true_test_prob')
    var_test_prob = load_single('./2nd_layer_data/only_var/true_test_prob')
    gene_test_prob = load_single('./2nd_layer_data/only_gene/true_test_prob')

    base_prob = np.zeros_like(xgb_test_prob)
    for i in range(base_prob.shape[0]):
        base_prob[i,:] = np.array([ 0.17103282,  0.13610358,  0.02679916,  0.20656429,  0.07286962, 0.08280638,  0.28696176,  0.00572117,  0.01114122])


    #predictions = [xgb_test_prob, w2v_test_prob, var_test_prob, gene_test_prob]
    predictions = [xgb_test_prob, tfidf_test_prob, w2v_test_prob, var_test_prob, gene_test_prob]
    X_ensemble = np.concatenate(predictions, axis = 1)
    helpers.submission('ensemble_submit', clf_ensemble.predict_proba(X_ensemble))
