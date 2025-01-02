# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 09:51:06 2023

@author: Jimmy
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class BasicIntraLink(tf.keras.layers.Layer):
    def __init__(self,units):
        super(BasicIntraLink,self).__init__()
        self.num = units
        self.layer1_1 = tf.keras.layers.Dense(units//2,activation='relu')
        self.layer1_2 = tf.keras.layers.Dense(units//2,)
        
    def build(self,input_shape):
        self.trainable_weight = self.add_weight(shape=(1,1),
                                  initializer= tf.keras.initializers.glorot_uniform(seed=123456), #'random_normal',
                                  trainable=True,
                                  name='w',
                                  )
        super(BasicIntraLink, self).build(input_shape)
    
    def call(self,inputs):
        x1_1 = self.layer1_1(inputs)
        x1_2 = tf.nn.relu(self.layer1_2(inputs)+self.trainable_weight*x1_1)
        output = tf.concat([x1_1,x1_2],axis=1)
        
        return output
    

class intra_link(tf.keras.Model):
    def __init__(self,num_blocks):
        super(intra_link,self).__init__()
        
        self.layer = self.build_model(8,num_blocks)
        self.fc = tf.keras.layers.Dense(1)

     
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
    def __init__(self,num_blocks):
        super(Normal_Model,self).__init__()
        
        self.layer = self.build_block_normal(8,num_blocks)
        
        self.fc = tf.keras.layers.Dense(1)
        
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




#### Helper functions ####
def my_data(x_min, x_max, n, train=True):
    x = np.linspace(x_min, x_max, n)
    x = np.expand_dims(x, -1).astype(np.float32)

    sigma = 0.5* np.ones_like(x) if train else np.zeros_like(x)
    y = x**2+x + np.random.normal(0, sigma).astype(np.float32)

    return x, y
            
# Create some training and testing data
x_train, y_train = my_data(-3, 3, 10000, train=True)
x_test, y_test = my_data(-4, 4, 1000, train=False)   


Ir = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.00001, cooldown=0, min_lr=0)
callback_lists=[Ir]

num_layers_intra = 3
tf.compat.v1.set_random_seed(1234) 
model_intra_link =  intra_link(num_layers_intra)
model_intra_link.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='mse')
history = model_intra_link.fit(x_train,y_train, batch_size=128, epochs=300,
                                callbacks=callback_lists)
y_pre_IntraLink = model_intra_link.predict(x_test)
mse_IntraLink = model_intra_link.evaluate(x_test,y_test)


num_layers_normal = 3
tf.compat.v1.set_random_seed(1234) 
model_normal = Normal_Model(num_layers_normal)       
model_normal.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='mse')
history = model_normal.fit(x_train,y_train, batch_size=128, epochs=300,
                            callbacks=callback_lists)
y_pre_Normal = model_normal.predict(x_test)
mse_normal = model_normal.evaluate(x_test,y_test)


    


