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
print("CV dim: ", m,n)
m,n = data_te.shape
data_te = data_te.reshape([m,n,1,1])
print("Test dim: ", m,n)
m,n = data_tr.shape
data_tr = data_tr.reshape([m,n,1,1])
print("Train dim: ", m,n)

batchsz = 100

indices = random.sample(range(0, m), batchsz)
x_tr = data_tr[indices] #[data[v] for v in indices]
y_tr = label_tr[indices] #[labels[v] for v in indices]

# general parameters
N = x_tr.shape[0] # number of training examples
D = x_tr.shape[1] # dimensionality of the data
C = y_tr.shape[1] # number of unique labels in the dataset
print("Batch sz?: ", N)
print("Example dim: ", D)
print("Unique classes: ", C)

# hyperparameters
L = 64 # lstm output size
H1 = 1024 # hidden 1 size
H2 = 256 # hidden 2 size

C1D = 630 # filter size
NC1 = 1 # number of channels
C2D = 87 # filter size
NC2 = 1 # number of channels
P = 4 # number of max pooling * pooling window size
lr = .0001 # the learning rate

#weights and initialization 

X = tf.placeholder("float", [None,D,1,1])
Y = tf.placeholder("float", [None, C])
W1 = tf.Variable(tf.truncated_normal([C1D,1,1,NC1], stddev=0.001))
b1 = tf.Variable(tf.truncated_normal([NC1],stddev=0.001))
W2 = tf.Variable(tf.truncated_normal([C2D,1,1,NC2], stddev=0.001))
b2 = tf.Variable(tf.truncated_normal([NC2],stddev=0.001))

# Fully Connected feed-forward
W_h1 = tf.Variable(tf.truncated_normal([L,H1], stddev = 0.01)) # mean=0.0
W_h2 = tf.Variable(tf.truncated_normal([H1,H2], stddev = 0.01)) # mean=0.0
W_o = tf.Variable(tf.truncated_normal([H2,C], stddev = 0.01)) # mean=0.0

b_h1 = tf.Variable(tf.zeros((1, H1)))
b_h2 = tf.Variable(tf.zeros((1, H2)))
b_o = tf.Variable(tf.zeros((1, C)))

# Convolution 1
C1_l = tf.add(tf.nn.conv2d(X, W1, [1,10,1,1], padding='VALID'), b1)
C1_a = tf.nn.relu(C1_l)
C1_mp = tf.nn.max_pool(C1_a, ksize = [1,4,1,1], strides=[1,4,1,1], padding='VALID')
C1_d = tf.nn.dropout(C1_a, 0.8)

C2_l = tf.add(tf.nn.conv2d(C1_d, W2, [1,10,1,1], padding='VALID'), b2)
C2_a = tf.nn.relu(C2_l)
C2_mp = tf.nn.max_pool(C2_a, ksize = [1,4,1,1], strides=[1,4,1,1], padding='VALID')
C2_d = tf.nn.dropout(C2_a, 0.8)

C2_out = tf.reshape(C2_d, [-1, 427,1])

# num_layers = 10
# state_size = 256
# # state_placeholder = tf.zeros(tf.float32, shape=[num_layers, 2, batchsz, state_size])
# # l = tf.unpack(state_placeholder, axis=0)
# # rnn_tuple_state = tuple(
# #          [tf.nn.rnn_cell.LSTMStateTuple(l[idx][0],l[idx][1])
# #           for idx in range(num_layers)]
# # )
# cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
# cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

# initial_state = state = cell.zero_state(batchsz, tf.float32)

# # lstms = [tf.nn.rnn_cell.GRUCell(size) for size in [128,64]+[L for i in range(8)]]
# # drops = [tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=0.8) for lstm in lstms]
# # cell = tf.nn.rnn_cell.MultiRNNCell(drops)


# lstm_out, final_state = tf.nn.dynamic_rnn(cell,C2_out, initial_state=initial_state)

# # states_series = tf.reshape(states_series, [-1, state_size])

# # gru1 = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(32)] * 10)
# # gru1_out, state = tf.nn.dynamic_rnn (gru1, C1_out, dtype=tf.float32, scope='gru1')
# lstm_out = lstm_out[:, -1, :]
# #lstm_out = tf.reshape(lstm_out, [-1,1087])

###############################################################################

num_layers = 12
hidden = L

init_state = tf.placeholder(tf.float32, [num_layers, 2, batchsz, hidden])
l = tf.unstack(init_state, axis=0)
rnn_tuple_state = tuple(
         [tf.nn.rnn_cell.LSTMStateTuple(l[idx][0],l[idx][1])
          for idx in range(num_layers)]
)
cell = tf.nn.rnn_cell.LSTMCell(hidden, forget_bias=.8)
cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

output, state = tf.nn.dynamic_rnn(cell, C2_out, dtype=tf.float32, initial_state=rnn_tuple_state)
output_end = output[:,-1,:]
output = tf.reshape(output, [-1,hidden])



###############################################################################

h1 = tf.nn.relu(tf.add(tf.matmul(output_end,W_h1), b_h1))
h2 = tf.nn.relu(tf.add(tf.matmul(h1,W_h2), b_h2))
y_hat = tf.nn.softmax(tf.add(tf.matmul(h2, W_o), b_o))

###############################################################################

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( 
                        labels=Y, logits=y_hat)) 

GD_step = tf.train.AdamOptimizer(lr).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    curstate = np.zeros((num_layers, 2, batchsz, hidden))+.0001
    curr_loss = sess.run(loss, feed_dict={X: data_te, Y: label_te, init_state: curstate})
    print ("The initial loss is: ", curr_loss)

    epoch = batchsz
    nepochs = 6
    epoch_size = int(data_tr.shape[0] / epoch)

    for i in trange(nepochs):
        r = np.random.permutation(data_tr.shape[0])
        for j in trange(epoch_size):
            indices = r[j*epoch:(j+1)*epoch]
            x_tr = data_tr[indices] #[data[v] for v in indices]
            y_tr = label_tr[indices] #[labels[v] for v in indices]

            _, l, cur_loss, curstate = sess.run([GD_step, y_hat, loss, state], feed_dict={X: x_tr, Y: y_tr, init_state: curstate})
            eval_accuracy = tf.equal(tf.argmax(l, 1), tf.argmax(y_tr, 1))
            print(" Iter accuracy: ", tf.reduce_mean(tf.cast(eval_accuracy, tf.float32)).eval())
            print("Training Loss: ", cur_loss)
            curr_loss, pred = sess.run([loss, y_hat], feed_dict={X: data_te, Y: label_te, init_state: curstate})
            print()
            print ("The final loss is: ", curr_loss)
            correctly_predicted = tf.equal(tf.argmax(pred, 1), tf.argmax(label_te, 1)) 
            print('argmax accuracy:', tf.reduce_mean(tf.cast(correctly_predicted, tf.float32)).eval())










###############################################################################

# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( 
#                         labels=Y, logits=y_hat)) 

# GD_step = tf.train.AdamOptimizer(lr).minimize(loss)

# correct_pred = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y_tr, 1))

# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())

# for i in range(8):
#     print("epoch: ", i)
#     _, accuracy_val = sess.run([GD_step, accuracy], feed_dict={X: x_tr, Y: y_tr})
#     print("loss: ", loss)

# print("shape: ", x_tr.shape)

# sess.run(tf.global_variables_initializer())

# iters = 8
# batch = 100

# for i in range(iters):
#     indices = random.sample(range(0, m), batch)
#     x_tr = data_tr[indices] #[data[v] for v in indices]
#     y_tr = label_tr[indices] #[labels[v] for v in indices]
#     GD_step.run(feed_dict={x:x_tr, y:y_tr})

#     correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#     print("Simple model accuracy:", accuracy.eval(feed_dict={x: test, y_: _y_test}))
# sess = tf.Session()
# init = tf.global_variables_initializer()
# sess.run(init)

# with tf.Session() as sess:

#     sess.run(tf.global_variables_initializer())

#     curr_loss = sess.run(loss, feed_dict={X: x_tr, Y: y_tr})
#     print ("Initial batch loss is: ", curr_loss)

#     sess.run(GD_step, feed_dict={X: x_tr, Y: y_tr})

#     nepochs = 8
#     epoch = 100
#     epoch_size = int(data_tr.shape[0] / epoch)

#     for i in range(nepochs):
#         m = 0
#         r = np.random.permutation(data_tr.shape[0])
#         for j in trange(epoch_size):

#             state = sess.run(initial_state)

#             indices = r[j*epoch:(j+1)*epoch]
#             x_tr = data_tr[indices] #[data[v] for v in indices]
#             y_tr = label_tr[indices] #[labels[v] for v in indices]

#             _, l = sess.run([GD_step, y_hat], feed_dict={X: x_tr, Y: y_tr})
#             eval_accuracy = tf.equal(tf.argmax(l, 1), tf.argmax(y_tr, 1))
#             m += tf.reduce_mean(tf.cast(eval_accuracy, tf.float32)).eval()
#         print("Training", m)

#     curr_loss, pred = sess.run([loss, y_hat], feed_dict={X: data_te, Y: label_te})
#     print()
#     print ("The final training loss is: ", curr_loss)
#     correctly_predicted = tf.equal(tf.argmax(pred, 1), tf.argmax(label_te, 1)) 
#     print('argmax accuracy:', tf.reduce_mean(tf.cast(correctly_predicted, tf.float32)).eval())