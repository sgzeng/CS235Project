from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import optimizers


class Network:

    def __init__(self, input_shape, num_classes, x_train, y_train,x_test,y_test, shuffle_ratio):

        self.model = Sequential()

        self.input_shape = input_shape
        self.num_classes = num_classes

        self.x_train = x_train
        self.x_test=x_test
        self.y_train = []
        self.y_test=[]
        self.y_clean=[]
        for i in range(0, len(y_train)):
            self.y_train.append(y_train[i])
            self.y_clean.append(y_train[i])
        for i in range(0,len(y_test)):
            self.y_test.append(y_test[i])

        self.shuffle(shuffle_ratio)
        self.y_train = keras.utils.to_categorical(self.y_train, self.num_classes)
        self.y_test = keras.utils.to_categorical(self.y_test, self.num_classes)
        # self.y_clean = keras.utils.to_categorical(self.y_clean, self.num_classes)

        self.y_test=np.array(self.y_test)
        self.y_train=np.array(self.y_train)
        self.y_clean=np.array(self.y_clean)

        k=1000
        self.model.add(Dense(k,activation='relu',input_shape=input_shape,
                             #kernel_initializer='he_normal'))
                              kernel_initializer=keras.initializers.RandomNormal(0,1.0)))
        # self.model.add(Dense(k, activation='relu', input_shape=input_shape
        #                      , kernel_initializer=keras.initializers.RandomNormal(0, 1.0)))
        self.model.add(Dense(num_classes,activation='softmax',
                             #kernel_initializer='he_normal'))
                             kernel_initializer=keras.initializers.RandomNormal(0,1.0/float(k))))
        #self.model.add(Dense(num_classes, activation='softmax'))

        self.model.compile(loss='cross_entropy',
                           optimizer='sgd', metrics=["accuracy"])
        #print self.model.summary()


    def shuffle(self, shuffle_ratio):
        import random

        self.shuffled_sample = random.sample(range(0, len(self.y_train)), int(len(self.y_train) * shuffle_ratio))
        for i in self.shuffled_sample:
            self.y_train[i] = random.randint(0, self.num_classes-1)*2-1
            # ori=self.y_train[i]
            # while abs(self.y_train[i]-ori)<=1:
            #     self.y_train[i] = random.randint(0, self.num_classes)
        #print self.y_train
        #print np.array(self.y_clean)-np.array(self.y_train),np.linalg.norm(np.array(self.y_clean)/2-np.array(self.y_train)/2)**2



    def set_weights(self, weight):
        self.model.set_weights(weight)

    def get_weights(self):
        return self.model.get_weights()

    def fit(self, batch_size, epochs,verbose=1,callbacks=[]):
        self.model.fit(
            self.x_train, self.y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
            validation_data=(self.x_test,self.y_test),
        )


