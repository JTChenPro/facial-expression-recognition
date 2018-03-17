
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import random
import sys
import numpy as np  
import time
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD, RMSprop
#import pandas as pd  
def load_data(path='./fer2013/fer2013.csv'):
    a=time.time()
    Y = []
    X = []
    first = True
    for line in open(path):
        if first:
            first = False
        else:
            row = line.split(',')
            Y.append(int(row[0]))
            X.append([int(p) for p in row[1].split()])
    X, Y = np.array(X), np.array(Y)
    b=time.time()
    print("cost-time is"+str(b-a)+"s")
    return X,Y
def deal_data(X,Y):
    Y[Y>0]=Y[Y>0]-1
    Y = to_categorical(Y)
    X=X.reshape([-1,48,48,1])
    X = X / 255.
    train_x=X[0:34000]
    train_y=Y[0:34000]
    test_x=X[34000:35887]
    test_y=Y[34000:35887]
    return X,Y,train_x,train_y,test_x,test_y
def describe(X_shape, y_shape, batch_size, dropout, nb_epoch, conv_arch, dense):
    print (' X_train shape: ', X_shape )
    print (' y_train shape: ', y_shape)
    print ('      img size: ', X_shape[2], X_shape[3])
    print ('    batch size: ', batch_size)
    print ('      nb_epoch: ', nb_epoch)
    print ('       dropout: ', dropout)
    print ('conv architect: ', conv_arch)
    print ('neural network: ', dense)
def cnn_architecture(X_train, y_train, conv_arch=[(32,3),(64,3),(128,3)],
                    dense=[64,2], dropout=0.3, batch_size=128, nb_epoch=100, validation_split=0.02, patience=5, dirpath='./data/results/'):
    starttime = time.time()
    X_train = X_train.astype('float32')
    X_shape = X_train.shape
    y_shape = y_train.shape
    describe(X_shape, y_shape, batch_size, dropout, nb_epoch, conv_arch, dense)
    # model architecture:
    model = Sequential()
    model.add(Convolution2D(conv_arch[0][0], 3, 3, border_mode='same', activation='relu',input_shape=( X_train.shape[1], X_train.shape[2],1)))
    if (conv_arch[0][1]-1) != 0:
        for i in range(conv_arch[0][1]-1):
            model.add(Convolution2D(conv_arch[0][0], 3, 3, border_mode='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    if conv_arch[1][1] != 0:
        for i in range(conv_arch[1][1]):
            model.add(Convolution2D(conv_arch[1][0], 3, 3, border_mode='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    if conv_arch[2][1] != 0:
        for i in range(conv_arch[2][1]):
            model.add(Convolution2D(conv_arch[2][0], 3, 3, border_mode='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    if dense[1] != 0:
        for i in range(dense[1]):
            model.add(Dense(dense[0], activation='relu'))
            if dropout:
                model.add(Dropout(dropout))
    prediction = model.add(Dense(y_train.shape[1], activation='softmax'))

    # optimizer:
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # set callback:
    callbacks = []
    print ('Training....')
    '''without data augmentation'''
    hist = model.fit(X_train, y_train, nb_epoch=nb_epoch, batch_size=batch_size,
              validation_split=validation_split, callbacks=callbacks, shuffle=True, verbose=1)
    print ('          Done!')
    return model


# In[ ]:

#x,y=load_data()
#x,y,train_x,train_y,cx,cy=deal_data(x.y)
#model=cnn_architecture(train_x,train_y,conv_arch=[(32,3),(64,3),(128,3)],dense=[64,2],dropout=0.42,batch_size=128,nb_epoch=10)
#model.fit(train_x, train_y, epochs=5, batch_size=128,validation_data=[cx,cy], callbacks=[], shuffle=True, verbose=1)

# In[ ]:

#model.save('mymode'+str(int(time.time()/100000))+'.h5')

