import numpy as np

def loadall(filename=''):
    tmp = np.load(filename)
    x_tr = tmp['x_tr']
    y_tr = tmp['y_tr']
    x_te = tmp['x_te']
    y_te = tmp['y_te']
    x_cv = tmp['x_cv']
    y_cv = tmp['y_cv']
    return {'x_tr' : x_tr, 'y_tr' : y_tr,
            'x_te' : x_te, 'y_te' : y_te,
            'x_cv' : x_cv, 'y_cv' : y_cv, }

# print(loadall('melspects.npz')['x_tr'].shape)
