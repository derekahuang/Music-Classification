import numpy as np
import random
import math
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

training = np.load('gtzan/gtzan_tr.npy')
data_tr = np.delete(training, -1, 1)
label_tr = training[:,-1]
print(label_tr)

test = np.load('gtzan/gtzan_te.npy')
data_te = np.delete(test, -1, 1)
label_te = test[:,-1]

cv = np.load('gtzan/gtzan_cv.npy')
data_cv = np.delete(cv, -1, 1)
label_cv = cv[:,-1]

print('here1', data_tr.shape)
print(label_cv)

scaler = StandardScaler()
# Fit on training set only.
scaler.fit(data_tr)
# Apply transform to both the training set and the test set.
train_sc = scaler.transform(data_tr)
cv_sc = scaler.transform(data_cv)
test_sc = scaler.transform(data_te)

print('here2')

pca = PCA(n_components = 15, whiten = True)
pca.fit(train_sc)

train_pca = pca.transform(train_sc)
cv_pca = pca.transform(cv_sc)
test_pca = pca.transform(test_sc)

print(pca.n_components_)

neigh = KNeighborsClassifier(n_neighbors=10, weights='distance')
neigh.fit(train_pca, label_tr)

preds = neigh.predict(cv_pca)
acc = np.sum(preds == label_cv)
acc = acc / len(label_cv)
print('Accuracy is {}'.format(acc))
print(preds)

