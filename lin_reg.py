# import matplotlib
# matplotlib.use("TkAgg")
# from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import random
import math
from tqdm import tqdm, trange

training = np.load('gtzan/gtzan_tr.npy')
data_tr = np.delete(training, -1, 1)
label_tr = training[:,-1]

test = np.load('gtzan/gtzan_te.npy')
data_te = np.delete(test, -1, 1)
label_te = test[:,-1]

cv = np.load('gtzan/gtzan_cv.npy')
data_cv = np.delete(cv, -1, 1)
label_cv = test[:,-1]

temp = np.zeros((len(label_tr),10))
temp[np.arange(len(label_tr)),label_tr.astype(int)] = 1
label_tr = temp
temp = np.zeros((len(label_te),10))
temp[np.arange(len(label_te)),label_te.astype(int)] = 1
label_te = temp
temp = np.zeros((len(label_cv),10))
temp[np.arange(len(label_cv)),label_cv.astype(int)] = 1
label_cv = temp
del temp

indices = random.sample(range(0, data_tr.shape[0]), 100)

x_tr = data_tr[indices] #[data[v] for v in indices]
y_tr = label_tr[indices] #[labels[v] for v in indices]
print(y_tr.shape)

# general parameters
N = x_tr.shape[0] # number of training examples
D = x_tr.shape[1] # dimensionality of the data
C = 10 # number of unique labels in the dataset

# hyperparameters
epoch = 100
H1 = 2048 # number of hidden units. In general try to stick to a power of 2
H2 = 1024
H3 = 512
H4 = 256
H5 = 128
H6 = 64
lr = .0001 # the learning rate (previously refered to in the notes as alpha)

W_h1 = tf.Variable(tf.random_normal((D,H1), stddev = 0.01)) # mean=0.0
W_h2 = tf.Variable(tf.random_normal((H1,H2), stddev = 0.01)) # mean=0.0
W_h3 = tf.Variable(tf.random_normal((H2,H3), stddev = 0.01)) # mean=0.0
W_h4 = tf.Variable(tf.random_normal((H3,H4), stddev = 0.01)) # mean=0.0
W_h5 = tf.Variable(tf.random_normal((H4,H5), stddev = 0.01)) # mean=0.0
W_h6 = tf.Variable(tf.random_normal((H5,H6), stddev = 0.01)) # mean=0.0
W_o = tf.Variable(tf.random_normal((H6,C), stddev = 0.01)) # mean=0.0

b_h1 = tf.Variable(tf.zeros((1, H1)))
b_h2 = tf.Variable(tf.zeros((1, H2)))
b_h3 = tf.Variable(tf.zeros((1, H3)))
b_h4 = tf.Variable(tf.zeros((1, H4)))
b_h5 = tf.Variable(tf.zeros((1, H5)))
b_h6 = tf.Variable(tf.zeros((1, H6)))
b_o = tf.Variable(tf.zeros((1, C)))

X = tf.placeholder("float", shape=[None,D])
y = tf.placeholder("float", shape=[None,C])

h1 = tf.nn.relu(tf.matmul(X,W_h1) + b_h1)
h2 = tf.nn.relu(tf.matmul(h1,W_h2) + b_h2)
h3 = tf.nn.relu(tf.matmul(h2,W_h3) + b_h3)
h4 = tf.nn.relu(tf.nn.dropout(tf.matmul(h3,W_h4) + b_h4), .3)
h5 = tf.nn.relu(tf.matmul(h4,W_h5) + b_h5)
h6 = tf.nn.relu(tf.matmul(h5,W_h6) + b_h6)
y_hat = tf.nn.softmax(tf.matmul(h6, W_o) + b_o)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( 
                        labels=y, logits=y_hat)) 

GD_step = tf.train.AdamOptimizer(lr).minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
with sess.as_default():
	curr_loss = sess.run(loss, feed_dict={X: data_te, y: label_te})
	print ("The initial loss is: ", curr_loss)

	# sess.run(GD_step, feed_dict={X: x_tr, y: y_tr})

	nepochs = 50
	epoch_size = int(data_tr.shape[0] / epoch)
	for i in trange(nepochs):
		training_mean = 0
		r = np.random.permutation(data_tr.shape[0])
		for j in trange(epoch_size):
			indices = r[j*epoch:(j+1)*epoch]
			x_tr = data_tr[indices] #[data[v] for v in indices]
			y_tr = label_tr[indices] #[labels[v] for v in indices]

			_, train_pred = sess.run([GD_step, y_hat], feed_dict={X: x_tr, y: y_tr})
			train_loss = tf.equal(tf.argmax(train_pred, 1), tf.argmax(y_tr, 1))
			training_mean += tf.reduce_mean(tf.cast(train_loss, tf.float32)).eval()
		print("Training accuracy: ", training_mean / epoch_size)
		_, eval_pred = sess.run([GD_step, y_hat], feed_dict={X: data_cv, y: label_cv})
		eval_accuracy = tf.equal(tf.argmax(eval_pred, 1), tf.argmax(label_cv, 1))
		print("Eval accuracy: ", tf.reduce_mean(tf.cast(eval_accuracy, tf.float32)).eval())

	curr_loss, pred = sess.run([loss, y_hat], feed_dict={X: data_te, y: label_te})
	print()
	print ("The final training loss is: ", curr_loss)
	correctly_predicted = tf.equal(tf.argmax(pred, 1), tf.argmax(label_te, 1)) 
	print('argmax accuracy:', tf.reduce_mean(tf.cast(correctly_predicted, tf.float32)).eval())
                 
