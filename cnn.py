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

m,n = data_cv.shape
data_cv = data_cv.reshape([m,n,1,1])
m,n = data_te.shape
data_te = data_te.reshape([m,n,1,1])
m,n = data_tr.shape
data_tr = data_tr.reshape([m,n,1,1])

epoch = 500

indices = random.sample(range(0, m), epoch)

x_tr = data_tr[indices] #[data[v] for v in indices]
y_tr = label_tr[indices] #[labels[v] for v in indices]

# general parameters
N = x_tr.shape[0] # number of training examples
D = x_tr.shape[1] # dimensionality of the data
C = 10 # number of unique labels in the dataset

# hyperparameters

H1 = 2048 # number of hidden units. 
H2 = 1024
H3 = 512
H4 = 256
H5 = 128
H6 = 64

C1D = 1024 # filter size
NC1 = 16 # number of channels
C2D = 512 # filter size
NC2 = 16 # number of channels

P = 4 # number of max pooling * pooling window size

lr = .01 # the learning rate (previously refered to in the notes as alpha)

#weights and initialization 

X = tf.placeholder("float", [None,D,1,1])
Y = tf.placeholder("float", [None, C])
W1 = tf.Variable(tf.truncated_normal([C1D,1,1,NC1], stddev=0.001))
b1 = tf.Variable(tf.truncated_normal([NC1],stddev=0.001))
W2 = tf.Variable(tf.truncated_normal([C2D,1,NC1,NC2], stddev=0.001))
b2 = tf.Variable(tf.truncated_normal([NC2], stddev=0.001))

# Fully Connected feed-forward
W_h1 = tf.Variable(tf.truncated_normal([int((D/P)*NC2),H1], stddev = 0.01)) # mean=0.0
W_h2 = tf.Variable(tf.truncated_normal([H1,H2], stddev = 0.01)) # mean=0.0
W_h3 = tf.Variable(tf.truncated_normal([H2,H3], stddev = 0.01)) # mean=0.0
W_h4 = tf.Variable(tf.truncated_normal([H3,H4], stddev = 0.01)) # mean=0.0
W_h5 = tf.Variable(tf.truncated_normal([H4,H5], stddev = 0.01)) # mean=0.0
W_h6 = tf.Variable(tf.truncated_normal([H5,H6], stddev = 0.01)) # mean=0.0
W_o = tf.Variable(tf.truncated_normal([H6,C], stddev = 0.01)) # mean=0.0

b_h1 = tf.Variable(tf.zeros((1, H1)))
b_h2 = tf.Variable(tf.zeros((1, H2)))
b_h3 = tf.Variable(tf.zeros((1, H3)))
b_h4 = tf.Variable(tf.zeros((1, H4)))
b_h5 = tf.Variable(tf.zeros((1, H5)))
b_h6 = tf.Variable(tf.zeros((1, H6)))
b_o = tf.Variable(tf.zeros((1, C)))

# Convolution 1

C1_out = tf.nn.conv2d(X, W1, [1,1,1,1], padding='SAME')                 
C1_out += b1
C1_out = tf.nn.relu(C1_out)   

C1_out_mp = tf.nn.max_pool(C1_out, ksize = [1,2,1,1], strides=[1,2,1,1], padding='SAME')

# Convolution 2

C2_out = tf.nn.conv2d(C1_out_mp, W2, [1,1,1,1], padding='SAME')                                  
C2_out += b2
C2_out = tf.nn.relu(C2_out)  

# Max Pooling 2
C2_out_mp = tf.nn.max_pool(C2_out, ksize = [1,2,1,1], strides = [1,2,1,1], padding='SAME')        

# Flatten
C2_out_mp = tf.reshape(C2_out_mp,[-1, int((D/P)*NC2)])  

h1 = tf.nn.relu(tf.matmul(C2_out_mp,W_h1) + b_h1)
h2 = tf.nn.relu(tf.matmul(h1,W_h2) + b_h2)
h3 = tf.nn.relu(tf.matmul(h2,W_h3) + b_h3)
d1 = tf.nn.dropout(h3, .3)
h4 = tf.nn.relu(tf.matmul(d1,W_h4) + b_h4)
h5 = tf.nn.relu(tf.matmul(h4,W_h5) + b_h5)
h6 = tf.nn.relu(tf.matmul(h5,W_h6) + b_h6)
y_hat = tf.nn.softmax(tf.matmul(h6, W_o) + b_o)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( 
                        labels=Y, logits=y_hat)) 

GD_step = tf.train.AdamOptimizer(lr).minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

with sess.as_default():
	curr_loss = sess.run(loss, feed_dict={X: data_te, Y: label_te})
	print ("The initial loss is: ", curr_loss)

	sess.run(GD_step, feed_dict={X: x_tr, Y: y_tr})

	nepochs = 20
	r = np.random.permutation(data_tr.shape[0])
	epoch_size = int(data_tr.shape[0] / epoch)
	for i in trange(nepochs):
		m = 0
		
		for j in trange(10):
			indices = r[j*epoch:(j+1)*epoch]
			x_tr = data_tr[indices] #[data[v] for v in indices]
			y_tr = label_tr[indices] #[labels[v] for v in indices]

			_, l = sess.run([GD_step, y_hat], feed_dict={X: data_cv, Y: label_cv})
			eval_accuracy = tf.equal(tf.argmax(l, 1), tf.argmax(label_cv, 1))
			print("Iter accuracy: ", tf.reduce_mean(tf.cast(eval_accuracy, tf.float32)).eval())
	l = sess.run(loss, feed_dict={X: data_tr[indices], Y: label_tr[indices]})
	print("Training Loss: ", l)
	curr_loss, pred = sess.run([loss, y_hat], feed_dict={X: data_te, Y: label_te})
	print()
	print ("The final loss is: ", curr_loss)
	correctly_predicted = tf.equal(tf.argmax(pred, 1), tf.argmax(label_te, 1)) 
	print('argmax accuracy:', tf.reduce_mean(tf.cast(correctly_predicted, tf.float32)).eval())
                 
