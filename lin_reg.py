# import matplotlib
# matplotlib.use("TkAgg")
# from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import random
import math
from tqdm import tqdm, trange

data = np.load('samples.npy')
labels = np.load('labels.npy')#[:,0:1]

indices = random.sample(range(0, 35200), 100)

x_tr = data[indices] #[data[v] for v in indices]
y_tr = labels[indices] #[labels[v] for v in indices]

# general parameters
N = x_tr.shape[0] # number of training examples
D = x_tr.shape[1] # dimensionality of the data
C = y_tr.shape[1] # number of unique labels in the dataset

def equation(x):
	return 440 * (2 ** (1/float(12))) ** (x - 49)

#mapping from frequencies to indices 

# mapping = {}
# for i in range(1, 89):
# 	mapping[math.floor(equation(i))] = i

# for i in range(len(y_tr)):
# 	y_tr[i][0] = mapping[math.floor(y_tr[i][0])]

# hyperparameters
H1 = 2048 # number of hidden units. In general try to stick to a power of 2
H2 = 1024
H3 = 512
H4 = 256
H5 = 128
H6 = 64
lr = .00001 # the learning rate (previously refered to in the notes as alpha)

W_h1 = tf.Variable(tf.random_normal((D,H1), stddev = 0.01)) # mean=0.0
W_h2 = tf.Variable(tf.random_normal((H1,H2), stddev = 0.01)) # mean=0.0
W_h3 = tf.Variable(tf.random_normal((H2,H3), stddev = 0.01)) # mean=0.0
W_o = tf.Variable(tf.random_normal((H3,C), stddev = 0.01)) # mean=0.0

b_h1 = tf.Variable(tf.zeros((1, H1)))
b_h2 = tf.Variable(tf.zeros((1, H2)))
b_h3 = tf.Variable(tf.zeros((1, H3)))
b_o = tf.Variable(tf.zeros((1, C)))

X = tf.placeholder("float", shape=[None,D])
y = tf.placeholder("float", shape=[None,C])

h1 = tf.nn.relu(tf.matmul(X,W_h1) + b_h1)
h2 = tf.nn.relu(tf.matmul(h1,W_h2) + b_h2)
h3 = tf.nn.relu(tf.matmul(h2,W_h3) + b_h3)
y_hat = tf.nn.softmax(tf.matmul(h3, W_o) + b_o)

l = tf.square(tf.add(y_hat, -y))
loss = tf.reduce_mean(tf.add(l, l))

vector_loss = tf.reduce_mean(tf.add(l, l), 0)

GD_step = tf.train.AdamOptimizer(lr).minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

curr_loss = sess.run(loss, feed_dict={X: data, y: labels})
print ("The initial loss is: ", curr_loss)

sess.run(GD_step, feed_dict={X: x_tr, y: y_tr})

nepochs = 50
for i in trange(nepochs):
	r = np.random.permutation(35200)
	for j in trange(352):
		indices = r[j*100:(j+1)*100]
		x_tr = data[indices] #[data[v] for v in indices]
		y_tr = labels[indices] #[labels[v] for v in indices]

		sess.run(GD_step, feed_dict={X: x_tr, y: y_tr})

curr_loss = sess.run(loss, feed_dict={X: data, y: labels})
vector_loss = sess.run(vector_loss, feed_dict={X: data, y: labels})
print()
print("The vectorized loss is: ", vector_loss)

print ("The final training loss is: ", curr_loss)
                 
sess.close()