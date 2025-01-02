# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 16:58:25 2023

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
        
        self.layer = self.build_model(256,num_blocks)
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
        
        self.layer = self.build_block_normal(256,num_blocks)
        
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
    
  
#%%  Walmart
import pandas as pd

features=pd.read_csv("features.csv").drop(columns="IsHoliday")
stores=pd.read_csv("stores.csv")
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
dataset=train.merge(stores,how="left").merge(features,how="left")

dataset=dataset[dataset.Date >= '2011-11-11']
dataset=dataset.dropna()
dataset =pd.get_dummies(dataset, columns=["Type",'IsHoliday']) 
dataset['Month']=pd.to_datetime(dataset['Date']).dt.month  
dataset['Year']=pd.to_datetime(dataset['Date']).dt.year
dataset['Day']=pd.to_datetime(dataset['Date']).dt.day
dataset=dataset.drop(columns='Date')


train_y=dataset.Weekly_Sales.values
train_y=train_y.reshape(-1,1)
train_x=dataset.drop(columns='Weekly_Sales')

from sklearn.preprocessing import MinMaxScaler,scale,StandardScaler
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
# train_x = scale(train_x)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(train_x,train_y,test_size=0.1)

Ir = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.00001, cooldown=0, min_lr=0)
callback_lists=[Ir]

num_layers_intra = 8
tf.compat.v1.set_random_seed(1234) 
model_intra_link =  intra_link(num_layers_intra)
model_intra_link.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='mse')
history_IntraLink = model_intra_link.fit(X_train,Y_train, batch_size=128, epochs=300,
                                          validation_split=0.1,callbacks=callback_lists)
y_pre_IntraLink = model_intra_link.predict(X_test)
mse_IntraLink = model_intra_link.evaluate(X_test,Y_test)
mean_errror_IntraLink = np.mean(np.abs(y_pre_IntraLink-Y_test))


num_layers_normal = 8
tf.compat.v1.set_random_seed(1234) 
model_normal = Normal_Model(num_layers_normal)       
model_normal.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='mse')
history_Normal = model_normal.fit(X_train,Y_train, batch_size=128, epochs=300,validation_split=0.1,
                                  callbacks=callback_lists)
y_pre_Normal = model_normal.predict(X_test)
mse_Normal = model_normal.evaluate(X_test,Y_test)
mean_errror_Normal = np.mean(np.abs(y_pre_Normal-Y_test))


#%%  Boston housing
# from sklearn import datasets
# X_train,Y_train = datasets.load_boston(return_X_y=True)

# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler=MinMaxScaler(feature_range=(0,1))
# X_train = scaler.fit_transform(X_train)

# from sklearn.model_selection import train_test_split
# X_train,X_test,Y_train,Y_test = train_test_split(X_train,Y_train,test_size=0.1)


# Ir = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.00001, cooldown=0, min_lr=0)
# callback_lists=[Ir]

# num_layers_intra = 3
# tf.compat.v1.set_random_seed(1234)  
# model_intra_link =  intra_link(num_layers_intra)
# model_intra_link.compile(
#     optimizer=tf.keras.optimizers.Adam(),
#     loss='mse')
# history_IntraLink = model_intra_link.fit(X_train,Y_train, batch_size=128, epochs=300,
#                                           callbacks=callback_lists)
# y_pre_IntraLink = model_intra_link.predict(X_test)
# mse_IntraLink = model_intra_link.evaluate(X_test,Y_test)
# mean_errror_IntraLink = np.mean(np.abs(y_pre_IntraLink-Y_test))

# num_layers_normal = 3
# tf.compat.v1.set_random_seed(1234)  
# model_normal = Normal_Model(num_layers_normal)       
# model_normal.compile(
#     optimizer=tf.keras.optimizers.Adam(),
#     loss='mse')
# history_Normal = model_normal.fit(X_train,Y_train, batch_size=128, epochs=300,
#                                   callbacks=callback_lists)
# y_pre_Normal = model_normal.predict(X_test)
# mse_Normal = model_normal.evaluate(X_test,Y_test)
# mean_errror_Normal = np.mean(np.abs(y_pre_Normal-Y_test))


#%% California housing
# from sklearn import datasets
# X_train,Y_train = datasets.fetch_california_housing(return_X_y=True)

# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler=MinMaxScaler(feature_range=(0,1))
# X_train = scaler.fit_transform(X_train)

# from sklearn.model_selection import train_test_split
# X_train,X_test,Y_train,Y_test = train_test_split(X_train,Y_train,test_size=0.1)


# Ir = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.00001, cooldown=0, min_lr=0)
# callback_lists=[Ir]


# num_layers_intra = 4
# tf.compat.v1.set_random_seed(1234)  
# model_intra_link =  intra_link(num_layers_intra)
# model_intra_link.compile(
#     optimizer=tf.keras.optimizers.Adam(),
#     loss='mse')
# history_IntraLink = model_intra_link.fit(X_train,Y_train, batch_size=128, epochs=300,
#                                           validation_split=0.1,callbacks=callback_lists)
# y_pre_IntraLink = model_intra_link.predict(X_test)
# mse_IntraLink = model_intra_link.evaluate(X_test,Y_test)
# mean_errror_IntraLink = np.mean(np.abs(y_pre_IntraLink-Y_test))

# num_layers_normal = 4
# tf.compat.v1.set_random_seed(1234)  
# model_normal = Normal_Model(num_layers_normal)       
# model_normal.compile(
#     optimizer=tf.keras.optimizers.Adam(),
#     loss='mse')
# history_Normal = model_normal.fit(X_train,Y_train, batch_size=128, epochs=300,
#                                   validation_split=0.1,callbacks=callback_lists)
# y_pre_Normal = model_normal.predict(X_test)
# mse_Normal = model_normal.evaluate(X_test,Y_test)
# mean_errror_Normal = np.mean(np.abs(y_pre_Normal-Y_test))


#%% Energy consumption
# import pandas as pd
# import datetime
# import numpy as np

# #data1 load
# data_load = pd.read_csv('energy_dataset.csv')
# data_load = data_load[['time','total load actual']]

# #process time
# time_ = np.array(data_load.time)

# for i in range (len(time_)):
#     a = str(time_[i:i+1])[2:21]
#     time_new_str = datetime.datetime.fromisoformat(a)
#     data_load.time[i:i+1] = np.array(time_new_str)
    
# data_load['time'] = pd.to_datetime(data_load['time'], format='%Y-%m-%d %H:%M:%S')   
# data_load['year'] = data_load['time'].dt.year
# data_load['month'] = pd.to_datetime(data_load['time']).dt.month
# data_load['day'] = pd.to_datetime(data_load['time']).dt.day
# data_load['hour'] = pd.to_datetime(data_load['time']).dt.hour
# data_load = data_load.drop(columns='time')

# # data2 load 
# data_weather = pd.read_csv('weather_features.csv')
# data_weather = data_weather.loc[data_weather['city_name'] =='Valencia' ]

# #process time
# time_ = np.array(data_weather.dt_iso)

# for i in range (len(time_)):
#     a = str(time_[i:i+1])[2:21]
#     time_new_str = datetime.datetime.fromisoformat(a)
#     data_weather.dt_iso[i:i+1] = np.array(time_new_str)

# data_weather['dt_iso'] = pd.to_datetime(data_weather['dt_iso'], format='%Y-%m-%d %H:%M:%S')   
# data_weather['year'] = data_weather['dt_iso'].dt.year
# data_weather['month'] = data_weather['dt_iso'].dt.month
# data_weather['day'] = data_weather['dt_iso'].dt.day
# data_weather['hour'] = data_weather['dt_iso'].dt.hour
# data_weather = data_weather.drop(columns=['dt_iso','city_name','rain_1h','rain_3h','snow_3h','weather_main','weather_description','weather_icon','weather_id'])

# data_all = data_load.merge(data_weather,how="left")
# hour_count = data_all.groupby(['year','month','day'])['hour'].count()
# repeat = data_all.duplicated(subset=['year','month','day','hour'])
# data_all = data_all.drop_duplicates(subset=['year','month','day','hour'])
# data_all = data_all.dropna()

# data_all = data_all.drop(columns=['year','day','temp_min','temp_max','wind_deg'])
# X_train = data_all.iloc[:,1:7]
# Y_train = data_all.iloc[:,0:1]

# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler=MinMaxScaler(feature_range=(0,1))
# X_train = scaler.fit_transform(X_train)

# from sklearn.model_selection import train_test_split
# X_train,X_test,Y_train,Y_test = train_test_split(X_train,Y_train,test_size=0.1)


# Ir = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.00001, cooldown=0, min_lr=0)
# callback_lists=[Ir]

# num_layers_intra = 15
# tf.compat.v1.set_random_seed(1234)  
# model_intra_link =  intra_link(num_layers_intra)
# model_intra_link.compile(
#     optimizer=tf.keras.optimizers.Adam(),
#     loss='mse')
# history_IntraLink = model_intra_link.fit(X_train,Y_train, batch_size=128, epochs=500,
#                                           validation_split=0.1,callbacks=callback_lists)
# y_pre_IntraLink = model_intra_link.predict(X_test)
# mse_IntraLink = model_intra_link.evaluate(X_test,Y_test)
# mean_errror_IntraLink = np.mean(np.abs(y_pre_IntraLink-Y_test))

# num_layers_normal = 15
# tf.compat.v1.set_random_seed(1234)  
# model_normal = Normal_Model(num_layers_normal)       
# model_normal.compile(
#     optimizer=tf.keras.optimizers.Adam(),
#     loss='mse')
# history_Normal = model_normal.fit(X_train,Y_train, batch_size=128, epochs=500,
#                                   validation_split=0.1,callbacks=callback_lists)
# y_pre_Normal = model_normal.predict(X_test)
# mse_Normal = model_normal.evaluate(X_test,Y_test)
# mean_errror_Normal = np.mean(np.abs(y_pre_Normal-Y_test))

