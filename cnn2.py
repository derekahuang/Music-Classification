# import matplotlib
# matplotlib.use("TkAgg")
# from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import random
import math
from tqdm import tqdm, trange

data = np.load('melspects.npz');

data_tr = data['x_tr']
label_tr = data['y_tr']
data_te = data['x_te']
label_te = data['y_te']
data_cv = data['x_cv']
label_cv = data['y_cv']
#training = np.load('gtzan/gtzan_tr.npy')
#data_tr = np.delete(training, -1, 1)
#label_tr = training[:,-1]
#
#test = np.load('gtzan/gtzan_te.npy')
#data_te = np.delete(test, -1, 1)
#label_te = test[:,-1]
#
#cv = np.load('gtzan/gtzan_cv.npy')
#data_cv = np.delete(cv, -1, 1)
#label_cv = test[:,-1]

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

m,n,l = data_cv.shape
data_cv = data_cv.reshape([m,n,l,1])
m,n,l = data_te.shape
data_te = data_te.reshape([m,n,l,1])
m,n,l = data_tr.shape
data_tr = data_tr.reshape([m,n,l,1])

epoch = 50

indices = random.sample(range(0, m), epoch)

x_tr = data_tr[indices] #[data[v] for v in indices]
y_tr = label_tr[indices] #[labels[v] for v in indices]

# general parameters
N = x_tr.shape[0] # number of training examples
D = x_tr.shape[1] # dimensionality of the data
C = 10 # number of unique labels in the dataset

# hyperparameters

H1 = 1000 # number of hidden units. 
H2 = 1024
H3 = 512
H4 = 256
H5 = 128
H6 = 80

C1D = 32 #filter size
NC1 = 32 #number of channels
C2D = 16# filter size
NC2 = 16 # number of channels
C3D = 8# filter size
NC3 = 8 # number of channels

P = 2 # number of max pooling * pooling window size

lr = .0001 # the learning rate (previously refered to in the notes as alpha)

#weights and initialization 

X = tf.placeholder("float", [None,D,173,1])
Y = tf.placeholder("float", [None, C])

W1 = tf.Variable(tf.truncated_normal([C1D,C1D,1,NC1], stddev=0.001))
b1 = tf.Variable(tf.truncated_normal([NC1],stddev=0.001))
W2 = tf.Variable(tf.truncated_normal([C2D,C2D,NC1,NC2], stddev=0.001))
b2 = tf.Variable(tf.truncated_normal([NC2], stddev=0.001))
W3 = tf.Variable(tf.truncated_normal([C3D,C3D,NC2,NC3], stddev=0.001))
b3 = tf.Variable(tf.truncated_normal([NC3], stddev=0.001))

# Fully Connected feed-forward
W_h1 = tf.Variable(tf.truncated_normal([int((D*173/P)*NC3),H1], stddev = 0.01)) # mean=0.0
W_h2 = tf.Variable(tf.truncated_normal([H1,H2], stddev = 0.01)) # mean=0.0
W_h3 = tf.Variable(tf.truncated_normal([H2,H3], stddev = 0.01)) # mean=0.0
W_h4 = tf.Variable(tf.truncated_normal([H1,H4], stddev = 0.01)) # mean=0.0
W_h5 = tf.Variable(tf.truncated_normal([H4,H5], stddev = 0.01)) # mean=0.0
W_h6 = tf.Variable(tf.truncated_normal([H1,H6], stddev = 0.01)) # mean=0.0
W_o = tf.Variable(tf.truncated_normal([H6,C], stddev = 0.01)) # mean=0.0

b_h1 = tf.Variable(tf.zeros((1, H1)))
b_h2 = tf.Variable(tf.zeros((1, H2)))
b_h3 = tf.Variable(tf.zeros((1, H3)))
b_h4 = tf.Variable(tf.zeros((1, H4)))
b_h5 = tf.Variable(tf.zeros((1, H5)))
b_h6 = tf.Variable(tf.zeros((1, H6)))
b_o = tf.Variable(tf.zeros((1, C)))

# Convolution 1

C1_out = tf.nn.dropout(tf.nn.conv2d(X, W1, [1,1,1,1], padding='SAME'), 1)
C1_out += b1
C1_out = tf.nn.relu(C1_out)   

C1_out_mp = tf.nn.max_pool(C1_out, ksize = [1,4,1,1], strides=[1,2,1,1], padding='SAME')

# Convolution 2

C2_out = tf.nn.conv2d(C1_out_mp, W2, [1,1,1,1], padding='SAME')                                  
C2_out += b2
C2_out = tf.nn.relu(C2_out)  

# Max Pooling 2
C2_out_mp = tf.nn.avg_pool(C2_out, ksize = [1,4,1,1], strides = [1,2,1,1], padding='SAME')        

# Convolution 3

C3_out = tf.nn.conv2d(C2_out_mp, W3, [1,1,1,1], padding='SAME')                                  
C3_out += b3
C3_out = tf.nn.relu(C3_out)  

# Max Pooling 3
C2_out_mp = tf.nn.avg_pool(C2_out, ksize = [1,4,1,1], strides = [1,2,1,1], padding='SAME')        

# Flatten
C2_out_mp = tf.reshape(C2_out_mp,[-1, int((D*173/P)*NC3)])  

h1 = tf.nn.relu(tf.matmul(C2_out_mp,W_h1) + b_h1)
# h2 = tf.nn.relu(tf.matmul(h1,W_h2) + b_h2)
# h3 = tf.nn.relu(tf.matmul(h2,W_h3) + b_h3)
d1 = tf.nn.dropout(h1, 1)
# h4 = tf.nn.relu(tf.matmul(h1,W_h4) + b_h4)
# h5 = tf.nn.relu(tf.matmul(h4,W_h5) + b_h5)
h6 = tf.nn.relu(tf.matmul(h1,W_h6) + b_h6)
y_hat = (tf.matmul(h6, W_o) + b_o)
# y_hat = tf.matmul((h6, W_o) + b_o)
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(
  lr,                # Base learning rate.
  global_step,  # Current index into the dataset.
  3200,          # Decay step.
  0.10,                # Decay rate.
  staircase=True)
#printed = tf.Print(learning_rate, [learning_rate], "Learning Rate: ")
reg= tf.nn.l2_loss(W_h1) + tf.nn.l2_loss(W_h6)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( 
                        labels=Y, logits=y_hat) + .001 * reg)

GD_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

with sess.as_default():
	f = open('output_lr2.txt','w')
	print(data_te.shape)
	curr_loss = sess.run(loss, feed_dict={X: data_te, Y: label_te})
	print ("The initial loss is: ", curr_loss)

	nepochs = 1000
	epoch_size = int(data_tr.shape[0] / epoch)
	for i in trange(nepochs):
		r = np.random.permutation(data_tr.shape[0])
		for j in trange(epoch_size):
			indices = r[j*epoch:(j+1)*epoch]
			x_tr = data_tr[indices] #[data[v] for v in indices]
			y_tr = label_tr[indices] #[labels[v] for v in indices]

			_, l, cur_loss = sess.run([GD_step, y_hat, loss], feed_dict={X: x_tr, Y: y_tr})
		lear, curr_loss, pred = sess.run([learning_rate, loss, y_hat], feed_dict={X: data_te, Y: label_te})
		print()
		print("The learning rate is: ", lear)
		print ("The final loss is: ", curr_loss)
		f.write("The final loss is: " + str(curr_loss))
		correctly_predicted = tf.equal(tf.argmax(pred, 1), tf.argmax(label_te, 1)) 
		print('argmax accuracy:', tf.reduce_mean(tf.cast(correctly_predicted, tf.float32)).eval())
		f.write('argmax accuracy:'+str(tf.reduce_mean(tf.cast(correctly_predicted, tf.float32)).eval()))
		f.flush()
	f.close()
                 
