import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.utils import np_utils
from keras import regularizers
from keras.engine.topology import Layer
import load_data

###################################################################################################

# Models to be passed to Music_Genre_CNN

def metric(y_true, y_pred):
    return K.mean(K.equal(K.argmax(y_true, axis=1), K.argmax(y_pred, axis=1)))

def cnn(num_genres=10, input_shape=(64,173,1)):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(4, 4),
                     activation='relu', #kernel_regularizer=regularizers.l2(0.04),
                     input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 4)))
    model.add(Conv2D(64, (3, 5), activation='relu'
                    , kernel_regularizer=regularizers.l2(0.04)
                    ))
    model.add(MaxPooling2D(pool_size=(2, 4)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (2, 2), activation='relu'
        , kernel_regularizer=regularizers.l2(0.04)
        ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.04)))
    model.add(Dropout(0.1))
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.04)))
    model.add(Dropout(0.1))
    model.add(Dense(num_genres, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                  metrics=[metric])
    return(model)

###################################################################################################

# Main network thingy to train

class model(object):

    def __init__(self, ann_model):
        self.model = ann_model()

    def train_model(self, input_spectrograms, labels, cv=True,
                validation_spectrograms=None, validation_labels=None,
                small_batch_size=200, max_iteration=300, print_interval=1,
                test_x=None, test_y=None):

        validation_accuracy_list = []
        for it in range(max_iteration):

            # split training data into even batches
            m = len(input_spectrograms)
            batch_idx = np.random.permutation(m)
            num_batches = int(m / small_batch_size)

            train_x = input_spectrograms[batch_idx]
            train_y = labels[batch_idx]

            for batch in range(num_batches):

                x_batch = train_x[ batch*small_batch_size : (batch+1)*small_batch_size]
                y_batch = train_y[ batch*small_batch_size : (batch+1)*small_batch_size]
                # print("starting batch\t", batch, "\t Epoch:\t", it)
                self.model.train_on_batch(x_batch, y_batch)
                # if (batch+1) % 50 == 0:
                #     print("getting accuracy")
                #     training_accuracy = self.model.evaluate(train_x, train_y)
                #     print("Training accuracy is: ", training_accuracy)

            if cv:
                validation_accuracy = self.model.evaluate(validation_spectrograms, validation_labels)
                validation_accuracy_list.append(validation_accuracy[1])
            else:
                validation_accuracy = [-1.0, -1.0]

            if it % print_interval == 0:
                training_accuracy = self.model.evaluate(train_x, train_y)
                testing_accuracy = self.model.evaluate(test_x, test_y)
                print("\nTraining accuracy: %f\t Validation accuracy: %f\t Testing Accuracy: %f" %
                      (training_accuracy[1], validation_accuracy[1], testing_accuracy[1]))
                print("\nTraining loss: %f    \t Validation loss: %f    \t Testing Loss: %f \n" %
                      (training_accuracy[0], validation_accuracy[0], testing_accuracy[0]))
                print( )
        if cv:
            return np.asarray(validation_accuracy_list)

###################################################################################################

def main():

#################################################

# Data stuff

    data = load_data.loadall('melspects.npz')

    x_tr = data['x_tr']
    y_tr = data['y_tr']
    x_te = data['x_te']
    y_te = data['y_te']
    x_cv = data['x_cv']
    y_cv = data['y_cv']

    tr_idx = np.random.permutation(len(x_tr))
    te_idx = np.random.permutation(len(x_te))
    cv_idx = np.random.permutation(len(x_cv))

    x_tr = x_tr[tr_idx]
    y_tr = y_tr[tr_idx]
    x_te = x_te[te_idx]
    y_te = y_te[te_idx]
    x_cv = x_cv[cv_idx]
    y_cv = y_cv[cv_idx]

    x_tr = x_tr[:,:,:,np.newaxis]
    x_te = x_te[:,:,:,np.newaxis]
    x_cv = x_cv[:,:,:,np.newaxis]

    y_tr = np_utils.to_categorical(y_tr)
    y_te = np_utils.to_categorical(y_te)
    y_cv = np_utils.to_categorical(y_cv)

#################################################

    ann = model(cnn)

    for i in range(10):
        print(i)
        validation_accuracies = ann.train_model(x_tr, y_tr, cv=True,
                                                validation_spectrograms=x_cv,
                                                validation_labels=y_cv,
                                                test_x=x_te, test_y=y_te)
        diff = np.mean(validation_accuracies[-10:]) - np.mean(validation_accuracies[:10])
        if np.abs(diff) < 0.01:
            break

if __name__ == '__main__':
    main()








