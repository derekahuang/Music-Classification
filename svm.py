import numpy as np
import load_data
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


data = load_data.loadall('melspects.npz')
x_tr = data['x_tr']
y_tr = data['y_tr']
x_te = data['x_te']
y_te = data['y_te']
x_cv = data['x_cv']
y_cv = data['y_cv']

print('here1', x_tr.shape)

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

pca = PCA(n_components = 15)
pca.fit(train_sc)

train_pca = pca.transform(train_sc)
cv_pca = pca.transform(cv_sc)
test_pca = pca.transform(test_sc)

print(pca.n_components_)

classifier = svm.SVC(gamma='scale', verbose=True)
classifier.fit(train_pca, y_tr)

preds = classifier.predict(cv_pca)
acc = np.sum(preds == y_cv)
acc = acc / len(y_cv)
print('Accuracy is {}'.format(acc))
print(preds)
