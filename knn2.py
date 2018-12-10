import numpy as np
import random
import math
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

import load_data

PCA_TOGGLE = True

data = load_data.loadall('melspects.npz')
x_tr = data['x_tr']
y_tr = data['y_tr']
x_te = data['x_te']
y_te = data['y_te']
x_cv = data['x_cv']
y_cv = data['y_cv']

print('here1', x_tr.shape)
# print(y_cv)

x_tr = x_tr.reshape(x_tr.shape[0], x_tr.shape[1]*x_tr.shape[2])
x_cv = x_cv.reshape(x_cv.shape[0], x_cv.shape[1]*x_cv.shape[2])
x_te = x_te.reshape(x_te.shape[0], x_te.shape[1]*x_te.shape[2])

scaler = StandardScaler()
# Fit on training set only.
scaler.fit(x_tr)
# Apply transform to both the training set and the test set.
train_sc = scaler.transform(x_tr)
cv_sc = scaler.transform(x_cv)
test_sc = scaler.transform(x_te)

print('here2')

if PCA_TOGGLE == True:
	pca = PCA(n_components = 15)
	pca.fit(train_sc)

	train_pca = pca.transform(train_sc)
	cv_pca = pca.transform(cv_sc)
	test_pca = pca.transform(test_sc)

	print(pca.n_components_)

	neigh = KNeighborsClassifier(n_neighbors=5, weights='distance')
	neigh.fit(train_pca, y_tr)

	train_preds = neigh.predict(train_pca)
	train_acc = np.sum(train_preds == y_tr)
	train_acc = train_acc / len(y_tr)

	cv_preds = neigh.predict(cv_pca)
	cv_acc = np.sum(cv_preds == y_cv)
	cv_acc = cv_acc / len(y_cv)

	test_preds = neigh.predict(test_pca)
	test_acc = np.sum(test_preds == y_te)
	test_acc = test_acc / len(y_te)

	print('Train: ', train_acc, "\tCV: ", cv_acc, "\tTest: ", test_acc)
	# print(preds)

else:
	neigh2 = KNeighborsClassifier(n_neighbors=10, weights='distance')
	neigh2.fit(train_sc, y_tr)

	preds = neigh2.predict(cv_)
	acc = np.sum(preds == y_cv)
	acc = acc / len(y_cv)
	print('Accuracy is {}'.format(acc))
	print(preds)

