# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 16:57:49 2023
intralink 内外>resnet>conventional>=pure intra
@author: Jimmy
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,Sequential

from keras.datasets import mnist,cifar10,cifar100
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
tf.compat.v1.disable_eager_execution()

#%%
class BasicBlock_IntraLink(tf.keras.layers.Layer):
    def __init__(self,filter_num,stride=1):
        super(BasicBlock_IntraLink,self).__init__()
        
        self.num = filter_num
        self.conv1_1 = tf.keras.layers.Conv2D(filter_num//2, (3,3),strides=stride,padding='same',
                                              kernel_initializer=tf.keras.initializers.VarianceScaling(),
                                              kernel_regularizer=tf.keras.regularizers.l2(5e-4)) #kernel_initializer=tf.keras.initializers.VarianceScaling()
        self.bn1_1 = tf.keras.layers.BatchNormalization()
        self.activation1_1 = tf.keras.layers.Activation('relu')
        
        self.conv1_2 = tf.keras.layers.Conv2D(filter_num//2, (3,3),strides=stride,padding='same',
                                              kernel_initializer=tf.keras.initializers.VarianceScaling(),
                                              kernel_regularizer=tf.keras.regularizers.l2(5e-4))
        self.bn1_2 = tf.keras.layers.BatchNormalization()
        self.activation1_2 = tf.keras.layers.Activation('relu')
        
        self.max = tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(1,1),padding='same')
        
            
    def build(self,input_shape):
        self.trainable_weight = self.add_weight(shape=(1,1,self.num//2),
                                  initializer= tf.keras.initializers.glorot_uniform(), #'random_normal',
                                  trainable=True,
                                  name='w',
                                  )
        super(BasicBlock_IntraLink, self).build(input_shape)
        
    def call(self,inputs):
        x1_1 = self.conv1_1(inputs)
        x1_1 = self.bn1_1(x1_1)
        x1_1 = self.activation1_1(x1_1)
        
        
        x1_2 = tf.keras.layers.add([self.conv1_2(inputs), self.trainable_weight*x1_1])  
        x1_2 = self.bn1_2(x1_2)
        output = tf.concat([x1_1,x1_2],axis=3)
        
        output = self.activation1_2(output)

        return output
  
#%%
class IntraLink(keras.Model):
    def __init__(self,layer_dims,num_classes=100):
        super(IntraLink, self).__init__()
        # 预处理层
        self.stem=Sequential([
            layers.Conv2D(64,(3,3),strides=1,padding='same',
                          kernel_initializer=tf.keras.initializers.VarianceScaling(),
                          kernel_regularizer=tf.keras.regularizers.l2(5e-4)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
        ])
        #block
        self.layer1= self.build_resblock(64,layer_dims[0],stride=1)
        self.layer2 = self.build_resblock(128, layer_dims[1],stride=2) 
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)
        self.max4 = tf.keras.layers.GlobalAveragePooling2D()
        
        
        self.flatten = tf.keras.layers.Flatten()
        self.fc=layers.Dense(num_classes,'softmax',
                             ) 
        
        

    def call(self,input,training=None):
        x=self.stem(input)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.max4(x)

        x=self.fc(x)
        return x

    def build_resblock(self,filter_num,blocks,stride=1):
        strides = [stride]+ [1] * (blocks - 1)
        res_blocks= Sequential()
        for stride in strides:
            res_blocks.add(BasicBlock_IntraLink(filter_num,stride))
        return res_blocks
    
def IntraLink_model18():
    return  IntraLink([3,3,3,3])


#%%
class BasicBlock(layers.Layer):
    def __init__(self,filter_num,activation='relu',stride=1,):
        super(BasicBlock, self).__init__()
        self.num = filter_num
        self.conv1=layers.Conv2D(filter_num,(3,3),strides=stride,padding='same'
                                 ,kernel_initializer=tf.keras.initializers.VarianceScaling(),
                                 kernel_regularizer=tf.keras.regularizers.l2(5e-4))
        self.bn1=layers.BatchNormalization()
        
        self.relu1=tf.keras.layers.Activation('relu')
        self.relu2=tf.keras.layers.Activation('relu')

        self.conv2=layers.Conv2D(filter_num,(3,3),strides=(1,1),padding='same',
                                 kernel_initializer=tf.keras.initializers.VarianceScaling(),
                                 kernel_regularizer=tf.keras.regularizers.l2(5e-4))
        self.bn2 = layers.BatchNormalization()

        if stride!=1:
            self.downsample=Sequential()
            self.downsample.add(layers.Conv2D(filter_num,(1,1),strides=stride,padding='same',
                                     kernel_initializer=tf.keras.initializers.VarianceScaling(),
                                     kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
            self.downsample.add(layers.BatchNormalization())
        else:
            self.downsample=lambda x:x
            
        
    def call(self,input,training=None):
        out=self.conv1(input)
        out=self.bn1(out)
        out=self.relu1(out)

        out=self.conv2(out)
        out=self.bn2(out)

        identity=self.downsample(input)
        output=layers.add([out,identity])
        output=self.relu2(output)

        return output
    
class ResNet(keras.Model):
    def __init__(self,layer_dims,num_classes=100):
        super(ResNet, self).__init__()
        # 预处理层
        self.stem=Sequential([
            layers.Conv2D(64,(3,3),strides=(1,1),padding='same',
                          kernel_initializer=tf.keras.initializers.VarianceScaling(),
                          kernel_regularizer=tf.keras.regularizers.l2(5e-4)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
        ])
        # resblock
        self.layer1 = self.build_resblock(64,'relu',layer_dims[0],stride=1)
        self.layer2 = self.build_resblock(128,'relu', layer_dims[1],stride=2) #256
        self.layer3 = self.build_resblock(256,'relu', layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, 'relu',layer_dims[3], stride=2)
        self.max4 = tf.keras.layers.GlobalAveragePooling2D()
        
        self.flatten = tf.keras.layers.Flatten()
        self.fc=layers.Dense(num_classes,'softmax',)



    def call(self,input,training=None):
        x=self.stem(input)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.max4(x)
        x=self.fc(x)
        return x

    def build_resblock(self,filter_num,act,blocks,stride=1):
        strides = [stride]+ [1] * (blocks - 1)
        res_blocks= Sequential()       
        for stride in strides:
            res_blocks.add(BasicBlock(filter_num,act,stride))
        return res_blocks
def resnet18():
    return  ResNet([2,2,2,2])


#%%
def compute_mean_var(image):
    mean = []
    var  = []
    for c in range(image.shape[-1]):
        mean.append(np.mean(image[:, :, :, c]))
        var.append(np.std(image[:, :, :, c]))
    return mean, var

#%%
import keras.backend as K

def stats_graph(graph):
    flops = tf.compat.v1.profiler.profile(graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
    params = tf.compat.v1.profiler.profile(graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))
 
 
def get_flops(model):
    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.compat.v1.profiler.profile(graph=tf.compat.v1.keras.backend.get_session().graph, run_meta=run_meta, cmd='op', options=opts)
    return flops.total_float_ops  # Prints the "flops" of the model.




#%%    
(trainX, trainY), (testX, testY) = cifar100.load_data()

trainX = trainX.astype('float32')
testX = testX.astype('float32')

trainX /= 255.0
testX /= 255.0
CIFAR100_TRAIN_MEAN,CIFAR100_TRAIN_STD = compute_mean_var(trainX)

trainX[:, :, :, 0] = (trainX[:, :, :, 0] - CIFAR100_TRAIN_MEAN[0]) / CIFAR100_TRAIN_STD[0]
trainX[:, :, :, 1] = (trainX[:, :, :, 1] - CIFAR100_TRAIN_MEAN[1]) / CIFAR100_TRAIN_STD[1]
trainX[:, :, :, 2] = (trainX[:, :, :, 2] - CIFAR100_TRAIN_MEAN[2]) / CIFAR100_TRAIN_STD[2]
testX[:, :, :, 0] = (testX[:, :, :, 0] - CIFAR100_TRAIN_MEAN[0]) / CIFAR100_TRAIN_STD[0]
testX[:, :, :, 1] = (testX[:, :, :, 1] - CIFAR100_TRAIN_MEAN[1]) / CIFAR100_TRAIN_STD[1]
testX[:, :, :, 2] = (testX[:, :, :, 2] - CIFAR100_TRAIN_MEAN[2]) / CIFAR100_TRAIN_STD[2]


trainY = tf.keras.utils.to_categorical(trainY, 100)
testY =  tf.keras.utils.to_categorical(testY, 100)

data_generate = ImageDataGenerator( 
                                    # rescale=1./255,
                                    featurewise_center=False,  # 将输入数据的均值设置为0
                                    samplewise_center=False,  # 将每个样本的均值设置为0
                                    featurewise_std_normalization=False,  # 将输入除以数据标准差，逐特征进行
                                    samplewise_std_normalization=False,  # 将每个输出除以其标准差
                                    zca_epsilon=1e-6,  # ZCA白化的epsilon值，默认为1e-6
                                    zca_whitening=False,  # 是否应用ZCA白化
                                    rotation_range=20,  # 随机旋转的度数范围，输入为整数
                                    width_shift_range=0.2,  # 左右平移，输入为浮点数，大于1时输出为像素值
                                    height_shift_range=0.2,  # 上下平移，输入为浮点数，大于1时输出为像素值
                                    shear_range=0.2,  # 剪切强度，输入为浮点数
                                    zoom_range=0.2,  # 随机缩放，输入为浮点数
                                    channel_shift_range=0.,  # 随机通道转换范围，输入为浮点数
                                    fill_mode='nearest',  # 输入边界以外点的填充方式，有 nearest,constant,reflect,wrap填充方式
                                    cval=0.,  # 用于填充的值，当fill_mode='constant'时生效
                                    horizontal_flip=True,  # 随机水平翻转
                                    vertical_flip=False,  # 随机垂直翻转
                                    rescale=None,  # 重随放因子，为None或0时不进行缩放
                                    preprocessing_function=None,  # 应用于每个输入的函数
                                    data_format=None,  # 图像数据格式，默认为channels_last
                                    validation_split=0.0)

train_generator = data_generate.flow(trainX,trainY,batch_size=128, shuffle=True)

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
Ir = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, 
                                          verbose=0, mode='auto', epsilon=0.00001, cooldown=0, min_lr=0)

callback_lists=[Ir]


vertical_resnet = IntraLink_model18()
tf.compat.v1.set_random_seed(12345) 
model_intralink = tf.keras.models.Sequential()
model_intralink.add(vertical_resnet)
model_intralink.compile( 
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.1,momentum=0.9,decay=5e-3),
              loss = 'categorical_crossentropy',
              metrics  = ['accuracy'])
history_intralink = model_intralink.fit(train_generator, epochs=100,
                    validation_data=(testX,testY),
                    verbose=1,callbacks=callback_lists)

loss_intralink,accuracy_intralink = model_intralink.evaluate(testX,testY)

# .... Define your model here .... FLOPs
print(get_flops(model_intralink))


H_resnet = resnet18()
tf.compat.v1.set_random_seed(12345)
model_resnet = tf.keras.models.Sequential()
model_resnet.add(H_resnet)
model_resnet.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.1,momentum=0.9,decay=5e-4),
              loss = 'categorical_crossentropy',
              metrics  = ['accuracy'])

history_resnet = model_resnet.fit(train_generator, epochs=100,
                    validation_data=(testX,testY),
                    verbose=1,callbacks=callback_lists)

loss_resnet,accuracy_resnet = model_resnet.evaluate(testX,testY)
# .... Define your model here ....
print(get_flops(model_resnet))
