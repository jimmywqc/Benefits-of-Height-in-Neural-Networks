# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 12:25:59 2023

@author: Jimmy
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class BasicIntraLink(tf.keras.layers.Layer):
    def __init__(self,units):
        super(BasicIntraLink,self).__init__()
        self.layer1_1 = tf.keras.layers.Dense(units//2,activation='relu')
        self.layer1_2 = tf.keras.layers.Dense(units//2,)
        
    def build(self,input_shape):
        self.trainable_alpha = self.add_weight(shape=(),
                                  initializer= tf.keras.initializers.glorot_uniform(seed=1234), #'random_normal',
                                  trainable=True,
                                  name='alpha',
                                  )
        super(BasicIntraLink, self).build(input_shape)
    
    def call(self,inputs):
        x1_1 = self.layer1_1(inputs)
        x1_2 = tf.nn.relu(self.layer1_2(inputs)+self.trainable_alpha*x1_1)
        output = tf.concat([x1_1,x1_2],axis=1)
        
        return output
    

class intra_link(tf.keras.Model):
    def __init__(self,num_blocks,num_class):
        super(intra_link,self).__init__()
        
        self.layer = self.build_model(8,num_blocks)
        self.fc = tf.keras.layers.Dense(num_class,'softmax')

     
    def call(self,inputs):
        x = self.layer(inputs)
        output = self.fc(x)       
        return output
    
    def build_model(self,units,num_blocks):
        block = tf.keras.Sequential()
        block.add(BasicIntraLink(units))
        for pre in range (1,num_blocks):
            block.add(BasicIntraLink(units))
        return block
        
    
class Normal_Model(tf.keras.Model):
    def __init__(self,num_blocks,num_class):
        super(Normal_Model,self).__init__()
        
        self.layer = self.build_block_normal(8,num_blocks)
        
        self.fc = tf.keras.layers.Dense(num_class,'softmax')
        
    def call(self,inputs):
        
        x = self.layer(inputs)   
        
        output = self.fc(x)
        return output
     
    def build_block_normal(self,units,num_blocks):
        block = tf.keras.Sequential()
        block.add(tf.keras.layers.Dense(units,'relu'))
        for pre in range (1,num_blocks):
            block.add(tf.keras.layers.Dense(units,'relu'))
        return block     
    
#%% datasets.make_gaussian_quantiles
from sklearn import datasets
X1, y1 = datasets.make_gaussian_quantiles(
    cov=2.0, n_samples=5000, n_features=2, n_classes=2, random_state=1
)
X2, y2 = datasets.make_gaussian_quantiles(
    mean=(3, 3), cov=1, n_samples=5000, n_features=2, n_classes=2, random_state=1
)
X_train = np.concatenate((X1, X2))
Y_train = np.concatenate((y1, -y2 + 1))
Y_train = Y_train.reshape(-1,1)

from sklearn.preprocessing import OneHotEncoder
OHE = OneHotEncoder()
Y_train = OHE.fit_transform(Y_train).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_train,Y_train,test_size=0.1)

Ir = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.00001, cooldown=0, min_lr=0)
callback_lists=[Ir]
num_class = 2

num_blocks_normal = 9
tf.compat.v1.set_random_seed(1234)  
model_normal = Normal_Model(num_blocks_normal,num_class)       
model_normal.compile(  loss        = 'categorical_crossentropy',
                optimizer   = 'adam',
                metrics     = ['accuracy']   
              )
history_normal = model_normal.fit(X_train,Y_train, batch_size=16, epochs=200,validation_split=0.1,
                            callbacks=callback_lists)
loss_normal,accuracy_normal = model_normal.evaluate(X_test, Y_test)
Y_hat_normal = model_normal.predict(X_test)
Y_hat_normal = np.argmax(Y_hat_normal,axis=1).reshape(-1,1)

num_blocks_intra = 8
tf.compat.v1.set_random_seed(1234)  
model_intra_link =  intra_link(num_blocks_intra,num_class)    
model_intra_link.compile(  loss        = 'categorical_crossentropy',
                optimizer   = 'adam',
                metrics     = ['accuracy']   
              )
history_intra_link = model_intra_link.fit(X_train,Y_train, batch_size=16, epochs=200, validation_split=0.1,
                                callbacks=callback_lists)
loss_intra_link,accuracy_intra_link = model_intra_link.evaluate(X_test, Y_test)
Y_hat_intra_link = model_intra_link.predict(X_test)
Y_hat_intra_link = np.argmax(Y_hat_intra_link,axis=1).reshape(-1,1)



