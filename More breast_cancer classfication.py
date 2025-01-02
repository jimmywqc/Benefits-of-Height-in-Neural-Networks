# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 20:48:00 2023

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
                                  initializer= tf.keras.initializers.glorot_uniform(seed=12345), #RandomUniform(minval=0, maxval=1), #'random_normal',
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
        
        self.layer = self.build_model(16,num_blocks)
        self.fc = tf.keras.layers.Dense(num_class,'sigmoid') #softmax

     
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
        
        self.layer = self.build_block_normal(16,num_blocks)
        
        self.fc = tf.keras.layers.Dense(num_class,'sigmoid') #softmax
        
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
    


#%%  breast_cancer using sigmoid

from sklearn import datasets
import pandas as pd

cancer_data_bunch = datasets.load_breast_cancer()
cancer_data = pd.DataFrame(cancer_data_bunch.data,columns=cancer_data_bunch.feature_names)
cancer_target = pd.DataFrame(cancer_data_bunch.target,columns=['target'])

X_train = cancer_data
Y_train = cancer_target

from sklearn.preprocessing import MinMaxScaler,scale,StandardScaler
# scaler = MinMaxScaler(feature_range=(0,1))
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

from sklearn.preprocessing import OneHotEncoder
OHE = OneHotEncoder()
Y_train = OHE.fit_transform(Y_train).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_train,Y_train,test_size=0.1)

Ir = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.00001, cooldown=0, min_lr=0)
callback_lists=[Ir]
num_class = 2

num_blocks_normal = 11
tf.compat.v1.set_random_seed(1234)  
model_normal = Normal_Model(num_blocks_normal,num_class)       
model_normal.compile(  loss        = 'categorical_crossentropy',
                optimizer   = 'adam',
                metrics     = ['accuracy']   
              )
history_normal = model_normal.fit(X_train,Y_train, batch_size=16, epochs=100,validation_split=0.1,
                            callbacks=callback_lists)
loss_normal,accuracy_normal = model_normal.evaluate(X_test, Y_test)
Y_hat_normal = model_normal.predict(X_test)
Y_hat_normal = np.argmax(Y_hat_normal,axis=1).reshape(-1,1)

num_blocks_intra = 10
tf.compat.v1.set_random_seed(1234)  
model_intra_link =  intra_link(num_blocks_intra,num_class)    
model_intra_link.compile(  loss        = 'categorical_crossentropy',
                optimizer   = 'adam',
                metrics     = ['accuracy']   
              )
history_intra_link = model_intra_link.fit(X_train,Y_train, batch_size=16, epochs=100, validation_split=0.1,
                                callbacks=callback_lists)
loss_intra_link,accuracy_intra_link = model_intra_link.evaluate(X_test, Y_test)
Y_hat_intra_link = model_intra_link.predict(X_test)
Y_hat_intra_link = np.argmax(Y_hat_intra_link,axis=1).reshape(-1,1)



#%%  fetch_kddcup99  

# from sklearn.datasets import fetch_kddcup99
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, OneHotEncoder,OrdinalEncoder
# from sklearn.model_selection import train_test_split
# #help(fetch_kddcup99)
# fetch_kddcup99_bunch = fetch_kddcup99()

# fetch_kddcup99_data = pd.DataFrame(fetch_kddcup99_bunch.data,columns=fetch_kddcup99_bunch.feature_names)
# fetch_kddcup99_target = pd.DataFrame(fetch_kddcup99_bunch.target,columns=fetch_kddcup99_bunch.target_names)

# fetch_kddcup99_data = OrdinalEncoder().fit_transform(fetch_kddcup99_data)
# fetch_kddcup99_data = MinMaxScaler(feature_range=(0,1)).fit_transform(fetch_kddcup99_data) 
# fetch_kddcup99_target = LabelEncoder().fit_transform(fetch_kddcup99_target).reshape(-1,1)

# OHE = OneHotEncoder()
# fetch_kddcup99_target = OHE.fit_transform(fetch_kddcup99_target).toarray()

# X_train, X_test, Y_train, Y_test = train_test_split(fetch_kddcup99_data,fetch_kddcup99_target,test_size=0.1)

# Ir = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.00001, cooldown=0, min_lr=0)
# callback_lists=[Ir]
# num_class = 23

# num_blocks_normal = 2
# tf.compat.v1.set_random_seed(1234)  
# model_normal = Normal_Model(num_blocks_normal,num_class)       
# model_normal.compile(  loss        = 'categorical_crossentropy',
#                 optimizer   = 'adam',
#                 metrics     = ['accuracy']   
#               )
# history_normal = model_normal.fit(X_train,Y_train, batch_size=64, epochs=100,validation_split=0.1,
#                             callbacks=callback_lists)
# loss_normal,accuracy_normal = model_normal.evaluate(X_test, Y_test)
# Y_hat_normal = model_normal.predict(X_test)
# Y_hat_normal = np.argmax(Y_hat_normal,axis=1).reshape(-1,1)

# num_blocks_intra = 2
# tf.compat.v1.set_random_seed(1234)  
# model_intra_link =  intra_link(num_blocks_intra,num_class)    
# model_intra_link.compile(  loss        = 'categorical_crossentropy',
#                 optimizer   = 'adam',
#                 metrics     = ['accuracy']   
#               )
# history_intra_link = model_intra_link.fit(X_train,Y_train, batch_size=64, epochs=100, validation_split=0.1,
#                                 callbacks=callback_lists)
# loss_intra_link,accuracy_intra_link = model_intra_link.evaluate(X_test, Y_test)
# Y_hat_intra_link = model_intra_link.predict(X_test)
# Y_hat_intra_link = np.argmax(Y_hat_intra_link,axis=1).reshape(-1,1)


#%% data_banknote_authentication using sigmoid

# import numpy as np
# import pandas as pd
# df = pd.read_csv('E:/CUHK/外部数据集/banknote+authentication/data_banknote_authentication.txt', sep=r'[,\t]')
# df.drop_duplicates(inplace=True)

# X_train = df.iloc[:,0:-1]
# Y_train = df.iloc[:,-1]
# Y_train = np.array(Y_train).reshape(-1,1)

# from sklearn.preprocessing import MinMaxScaler,scale,StandardScaler
# scaler = MinMaxScaler(feature_range=(0,1))
# # scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)

# from sklearn.preprocessing import OneHotEncoder
# OHE = OneHotEncoder()
# Y_train = OHE.fit_transform(Y_train).toarray()

# from sklearn.model_selection import train_test_split
# X_train, X_test, Y_train, Y_test = train_test_split(X_train,Y_train,test_size=0.1)

# Ir = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.00001, cooldown=0, min_lr=0)
# callback_lists=[Ir]

# num_class = 2

# num_blocks_normal = 1
# tf.compat.v1.set_random_seed(1234)  
# model_normal = Normal_Model(num_blocks_normal,num_class)       
# model_normal.compile(  loss        = 'categorical_crossentropy',
#                 optimizer   = 'adam',
#                 metrics     = ['accuracy']   
#               )
# history_normal = model_normal.fit(X_train,Y_train, batch_size=16, epochs=300,validation_split=0.1,
#                             callbacks=callback_lists)
# loss_normal,accuracy_normal = model_normal.evaluate(X_test, Y_test)
# Y_hat_normal = model_normal.predict(X_test)
# Y_hat_normal = np.argmax(Y_hat_normal,axis=1).reshape(-1,1)

# num_blocks_intra = 1
# tf.compat.v1.set_random_seed(1234)  
# model_intra_link =  intra_link(num_blocks_intra,num_class)    
# model_intra_link.compile(  loss        = 'categorical_crossentropy',
#                 optimizer   = 'adam',
#                 metrics     = ['accuracy']   
#               )
# history_intra_link = model_intra_link.fit(X_train,Y_train, batch_size=16, epochs=300, validation_split=0.1,
#                                 callbacks=callback_lists)
# loss_intra_link,accuracy_intra_link = model_intra_link.evaluate(X_test, Y_test)
# Y_hat_intra_link = model_intra_link.predict(X_test)
# Y_hat_intra_link = np.argmax(Y_hat_intra_link,axis=1).reshape(-1,1)



#%% ionosphere

# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# df = pd.read_csv('E:/CUHK/外部数据集/ionosphere/ionosphere.data',header = None,)
# df.drop_duplicates(inplace=True)
# # df = df.drop(axis=0,index=[9,217])

# X_train = df.iloc[:,0:-1]
# Y_train = df.iloc[:,-1]
# Y_train = np.array(Y_train).reshape(-1,1)

# from sklearn.preprocessing import MinMaxScaler,scale,StandardScaler
# # scaler = MinMaxScaler(feature_range=(-1,1))
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)

# from sklearn.preprocessing import OneHotEncoder
# OHE = OneHotEncoder()
# Y_train = OHE.fit_transform(Y_train).toarray()


# from sklearn.model_selection import train_test_split
# X_train, X_test, Y_train, Y_test = train_test_split(X_train,Y_train,test_size=0.1)

# Ir = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto',
#                                           epsilon=0.00001, cooldown=0, min_lr=0)
# early_stop = tf.keras.callbacks.EarlyStopping( monitor="val_accuracy",min_delta=0.0001, patience=5,
#     verbose=1,mode="auto", baseline=None,restore_best_weights=True
# )
# callback_lists=[Ir]
# num_class = 2

# num_blocks_normal = 5
# tf.compat.v1.set_random_seed(12345)  
# model_normal = Normal_Model(num_blocks_normal,num_class)       
# model_normal.compile(  loss        = 'binary_crossentropy',
#                 optimizer   = 'adam',
#                 metrics     = ['accuracy']   
#               )
# history_normal = model_normal.fit(X_train,Y_train, batch_size=16, epochs=100,validation_split=0.1,
#                             callbacks=callback_lists)
# loss_normal,accuracy_normal = model_normal.evaluate(X_test, Y_test)
# Y_hat_normal = model_normal.predict(X_test)
# Y_hat_normal = np.argmax(Y_hat_normal,axis=1).reshape(-1,1)

# num_blocks_intra = 4
# tf.compat.v1.set_random_seed(12345)  
# model_intra_link =  intra_link(num_blocks_intra,num_class)    
# model_intra_link.compile(  loss        = 'binary_crossentropy',
#                 optimizer   = 'adam',
#                 metrics     = ['accuracy']   
#               )
# history_intra_link = model_intra_link.fit(X_train,Y_train, batch_size=16, epochs=100, validation_split=0.1,
#                                 callbacks=callback_lists)
# loss_intra_link,accuracy_intra_link = model_intra_link.evaluate(X_test, Y_test)
# Y_hat_intra_link = model_intra_link.predict(X_test)
# Y_hat_intra_link = np.argmax(Y_hat_intra_link,axis=1).reshape(-1,1)



#%% Mobile Price 
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# df = pd.read_csv('E:/CUHK/外部数据集/Mobile Price Classification/train.csv',)
# df.drop_duplicates(inplace=True)

# X_train = df.iloc[:,0:-1]
# Y_train = df.iloc[:,-1]
# Y_train = np.array(Y_train).reshape(-1,1)

# from sklearn.preprocessing import MinMaxScaler,scale,StandardScaler
# # scaler = MinMaxScaler(feature_range=(-1,1))
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)

# from sklearn.preprocessing import OneHotEncoder
# OHE = OneHotEncoder()
# Y_train = OHE.fit_transform(Y_train).toarray()


# from sklearn.model_selection import train_test_split
# X_train, X_test, Y_train, Y_test = train_test_split(X_train,Y_train,test_size=0.1)

# Ir = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto',
#                                           epsilon=0.00001, cooldown=0, min_lr=0)
# early_stop = tf.keras.callbacks.EarlyStopping( monitor="val_accuracy",min_delta=0.0001, patience=5,
#     verbose=1,mode="auto", baseline=None,restore_best_weights=True
# )
# callback_lists=[Ir]

# num_class = 4

# num_blocks_normal = 4
# tf.compat.v1.set_random_seed(12345)  
# model_normal = Normal_Model(num_blocks_normal,num_class)       
# model_normal.compile(  loss        = 'categorical_crossentropy',
#                 optimizer   = 'adam',
#                 metrics     = ['accuracy']   
#               )
# history_normal = model_normal.fit(X_train,Y_train, batch_size=16, epochs=100,validation_split=0.1,
#                             callbacks=callback_lists)
# loss_normal,accuracy_normal = model_normal.evaluate(X_test, Y_test)
# Y_hat_normal = model_normal.predict(X_test)
# Y_hat_normal = np.argmax(Y_hat_normal,axis=1).reshape(-1,1)

# num_blocks_intra = 4
# tf.compat.v1.set_random_seed(12345)
# model_intra_link =  intra_link(num_blocks_intra,num_class)    
# model_intra_link.compile(  loss        = 'categorical_crossentropy',
#                 optimizer   = 'adam',
#                 metrics     = ['accuracy']   
#               )
# history_intra_link = model_intra_link.fit(X_train,Y_train, batch_size=16, epochs=100, validation_split=0.1,
#                                 callbacks=callback_lists)
# loss_intra_link,accuracy_intra_link = model_intra_link.evaluate(X_test, Y_test)
# Y_hat_intra_link = model_intra_link.predict(X_test)
# Y_hat_intra_link = np.argmax(Y_hat_intra_link,axis=1).reshape(-1,1)


#%% CWRU
# import scipy.io as scio
# from random import shuffle


# def cut_samples(org_signals):
#     ''' get original signals to 10*240*1024 samples, meanwhile normalize these samples
#     :param org_signals :a 10* 121048 matrix of ten original signals 
#     '''
#     inputs = []
#     labels = []
    
#     for i in range (10):
#         signal = org_signals[i]
#         for j in range (240):
#             labels.append(i)
#             inputs.append(signal[500*j:1024+500*j])
#             # inputs.append(signal[1000*j:2048+500*j]) #120
    
#     return inputs,labels

# dataFile= 'c10signals.mat'
# data=scio.loadmat(dataFile)
# org_signals=data['signals']
# inputs,labels=cut_samples(org_signals)

# X_train = np.array(inputs)
# Y_train = np.array(labels).reshape(-1,1)

# from sklearn.preprocessing import MinMaxScaler,scale,StandardScaler
# # scaler = MinMaxScaler(feature_range=(0,1))
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)

# from sklearn.preprocessing import OneHotEncoder#,LabelEncoder
# OHE = OneHotEncoder()
# Y_train = OHE.fit_transform(Y_train).toarray()

# from sklearn.model_selection import train_test_split
# X_train,X_test,Y_train,Y_test = train_test_split(X_train,Y_train,test_size=0.1)


# Ir = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto',
#                                           epsilon=0.00001, cooldown=0, min_lr=0)
# early_stop = tf.keras.callbacks.EarlyStopping( monitor="val_accuracy",min_delta=0.0001, patience=5,
#     verbose=1,mode="auto", baseline=None,restore_best_weights=True
# )
# callback_lists=[Ir]

# num_class = 10

# num_blocks_normal = 8
# tf.compat.v1.set_random_seed(12345)  
# model_normal = Normal_Model(num_blocks_normal,num_class)       
# model_normal.compile(  loss        = 'categorical_crossentropy',
#                 optimizer   = 'adam',
#                 metrics     = ['accuracy']   
#               )
# history_normal = model_normal.fit(X_train,Y_train, batch_size=16, epochs=100,validation_split=0.1,
#                             callbacks=callback_lists)
# loss_normal,accuracy_normal = model_normal.evaluate(X_test, Y_test)
# Y_hat_normal = model_normal.predict(X_test)
# Y_hat_normal = np.argmax(Y_hat_normal,axis=1).reshape(-1,1)

# num_blocks_intra = 7
# tf.compat.v1.set_random_seed(12345)  
# model_intra_link =  intra_link(num_blocks_intra,num_class)    
# model_intra_link.compile(  loss        = 'categorical_crossentropy',
#                 optimizer   = 'adam',
#                 metrics     = ['accuracy']   
#               )
# history_intra_link = model_intra_link.fit(X_train,Y_train, batch_size=16, epochs=100, validation_split=0.1,
#                                 callbacks=callback_lists)
# loss_intra_link,accuracy_intra_link = model_intra_link.evaluate(X_test, Y_test)
# Y_hat_intra_link = model_intra_link.predict(X_test)
# Y_hat_intra_link = np.argmax(Y_hat_intra_link,axis=1).reshape(-1,1)


