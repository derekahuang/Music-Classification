import tensorflow as tf
import tqdm
import numpy as np
import os


training = np.load('gtzan/gtzan_tr.npy')
data_tr = np.delete(training, -1, 1)
label_tr = training[:,-1]

test = np.load('gtzan/gtzan_te.npy')
data_te = np.delete(test, -1, 1)
label_te = test[:,-1]

cv = np.load('gtzan/gtzan_cv.npy')
data_cv = np.delete(cv, -1, 1)
label_cv = cv[:,-1]

m,n = data_cv.shape
data_cv = data_cv.reshape([m,n,1,1])
m,n = data_te.shape
data_te = data_te.reshape([m,n,1,1])
m,n = data_tr.shape
data_tr = data_tr.reshape([m,n,1,1])

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

print "Building the TF graph"

# data parameters
N = data_tr.shape[0] # total number of training datapoints
D = data_tr.shape[1] # dimensionality
print D
C = 10 # number of classes
C1D = 1024 # filter size
NC1 = 16 # number of channels
C2D = 512 # filter size
NC2 = 16 # number of channels
NH = 256 # hidden units (before softmax)
lr = 0.000001

# tensorflow placeholders and variables
X = tf.placeholder("float", [None,D,1,1])
Y = tf.placeholder("float", [None, C])
W1 = tf.Variable(tf.truncated_normal([C1D,1,1,NC1], stddev=0.001))
b1 = tf.Variable(tf.truncated_normal([NC1]))
W2 = tf.Variable(tf.truncated_normal([C2D,1,NC1,NC2], stddev=0.001))
b2 = tf.Variable(tf.truncated_normal([NC2]))
W3 = tf.Variable(tf.truncated_normal([(D/4)*NC2,NH], stddev=0.001))                 
b3 = tf.Variable(tf.truncated_normal([NH]))                 
W4 = tf.Variable(tf.truncated_normal([NH,C], stddev=0.001))                 
b4 = tf.Variable(tf.truncated_normal([C]))        

#### Forward pass ###

# Convolution 1                 
C1_out = tf.nn.conv2d(X, W1, [1,1,1,1], padding='SAME')                 
C1_out += b1
C1_out = tf.nn.relu(C1_out)                 

print C1_out

# Max Pooling 1
C1_out_mp = tf.nn.max_pool(C1_out, ksize = [1,2,1,1], strides=[1,2,1,1], padding='SAME')                 

print C1_out_mp
                 
# Convolution 2                 
C2_out = tf.nn.conv2d(C1_out_mp, W2, [1,1,1,1], padding='SAME')                                  
C2_out += b2
C2_out = tf.nn.relu(C2_out)                 

print C2_out

# Max Pooling 2
C2_out_mp = tf.nn.max_pool(C2_out, ksize = [1,2,1,1], strides = [1,2,1,1], padding='SAME')                 

print C2_out_mp

# # Fully connected 1
C2_out_mp = tf.reshape(C2_out_mp,[-1, (D/4)*NC2])                 
print C2_out_mp
H1 = tf.matmul(C2_out_mp, W3) + b3
H1 = tf.nn.relu(H1)
H1 = tf.nn.dropout(H1,0.90) # dropout
                 
# # Fully connected 2 (softmax)                 
scores = tf.matmul(H1, W4) + b4
y_hat = tf.nn.softmax(scores)

# # compute the loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=Y))                  

# # training rule
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
GD_step = optimizer.minimize(loss)

# run the session
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

print "Initiating Training ..."

nepochs = 100
nminibatch = 10
minibatchsize = int(N/nminibatch)
for i in xrange(nepochs):            
	
	r = np.random.permutation(data_tr.shape[0])

	for iminibatch in xrange(nminibatch):
		indices = r[iminibatch*minibatchsize:(iminibatch*minibatchsize+iminibatch)]
		x_tr = data_tr[indices] #[data[v] for v in indices]
		y_tr = label_tr[indices] #[labels[v] for v in indices]

		training_loss = sess.run(loss, feed_dict={X: x_tr, Y: y_tr})
		print "Epoch: ", i, ". Minibatch No. ", iminibatch, " of ",nminibatch, " total. Training loss: ", training_loss		
		# gradient descent    
		sess.run(GD_step, feed_dict={X: x_tr, Y: y_tr})
	

	# validation data assessment	
	validation_scores = sess.run(scores, feed_dict={X: data_te, Y: label_te})
	print "Validation accuracy: " , 100.0*np.sum(np.equal(np.argmax(validation_scores,axis=1),np.argmax(label_te,axis=1)))/label_te.shape[0]	
