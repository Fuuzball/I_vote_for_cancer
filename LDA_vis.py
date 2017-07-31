import sklearn.discriminant_analysis as LDA
import matplotlib.pylab as plt
import numpy as np

X_train = np.load('./data/wtf_X_train.npy')
X_test = np.load('./data/wtf_X_test.npy')

y_train = np.load('./data/wtf_y_train.npy')
y_test = np.load('./data/wtf_y_test.npy')

l_train = np.nonzero(y_train)[1]
l_test = np.nonzero(y_test)[1] 

lda = LDA.LinearDiscriminantAnalysis(n_components=2)
X_train_lda = lda.fit_transform(X_train, l_train)
X_test_lda = lda.fit_transform(X_test, l_test)

plt.figure()
plt.scatter(X_train_lda[:, 0], X_train_lda[:, 1], s=2, c=l_train, cmap='hsv')
plt.figure()
plt.scatter(X_test_lda[:, 0], X_test_lda[:, 1], s=2, c=l_test, cmap='hsv')
plt.show()
