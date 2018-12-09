import numpy as np
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf
from keras.utils import np_utils
from keras import regularizers
from keras.engine.topology import Layer
import load_data

###################################################################################################

def model(num_genres=num_genres, input_shape=input_shape):
    model = Sequential()
    model.add(Conv2D(128, kernel_size=(4, 4),
                     activation='relu', kernel_regularizer=regularizers.l2(0.02),
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 4)))
    model.add(Conv2D(64, (3, 5), activation='relu', kernel_regularizer=regularizers.l2(0.02)))
    model.add(MaxPooling2D(pool_size=(2, 4)))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.02)))
    model.add(Dropout(0.1))
    model.add(Dense(num_genres, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=1e-4),
                  metrics=['accuracy'])
    return(model)

###################################################################################################

def main():
    data = load_data.loadall('melspects.npz')

#################################################

# Data stuff

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

    y_tr = np_utils.to_categorical(y_tr)
    y_te = np_utils.to_categorical(y_te)
    y_cv = np_utils.to_categorical(y_cv)

#################################################




