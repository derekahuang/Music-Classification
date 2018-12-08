import load_data

data = load_data.loadall('melspects.npz')
x_tr = data['x_tr']
y_tr = data['y_tr']
x_te = data['x_te']
y_te = data['y_te']
x_cv = data['x_cv']
y_cv = data['y_cv']

print("x_tr:\t", x_tr.shape)
print("y_tr:\t", y_tr.shape)
print("x_te:\t", x_te.shape)
print("y_te:\t", y_te.shape)
print("x_cv:\t", x_cv.shape)
print("y_cv:\t", y_cv.shape)