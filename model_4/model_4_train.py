import tensorflow as tf 
import numpy as np
#import cv2
import matplotlib.pyplot as plt

from tensorflow.keras import Model, Sequential, layers, utils
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Activation,MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from collections import Counter
import time
import os

class D2CNN(Model):

    def __init__(self, *args, **kwargs):
        super(D2CNN,self).__init__()
        self.conv1 = Sequential([
        Conv2D(10,(11,11),strides=4,padding='same',name = 'conv1_1'),
        Activation("relu",name = 'act1'),
        BatchNormalization(name = 'batch_norm_1'),
        MaxPooling2D(pool_size=(3,3),strides = 2, name = 'pool_1')
        ])
        self.conv2 = Sequential([
        Conv2D(20,(5,5),strides=1,padding='same',name = 'conv2_1'),
        Activation("relu",name = 'act2'),
        BatchNormalization(name = 'batch_norm_2'),
        MaxPooling2D(pool_size=(2,2),strides = 2, name = 'pool_2')
        ])
        self.conv3 = Sequential([
        Conv2D(40,(5,5),strides=1,padding='same',name = 'conv3_1'),
        Activation("relu",name = 'act3'),
        BatchNormalization(name = 'batch_norm_3'),
        MaxPooling2D(pool_size=(2,2),strides = 2, name = 'pool_3')
        ])
#         self.conv4 = Sequential([
#         Conv2D(256,(5,5),strides=1,padding='same',name = 'conv4_1'),
#         Activation("relu",name = 'act4'),
#         BatchNormalization(name = 'batch_norm_4'),
#         MaxPooling2D(pool_size=(2,2),strides = 2, name = 'pool_4')
#         ])
        
        self.flatten = Flatten()
        
        self.classifier =  Sequential([
        Dense(1024,activation = 'relu'),
        layers.Dropout(0.5),
        Dense(1024,activation = 'relu'),
        layers.Dropout(0.5),
        Dense(10,activation='softmax')
        ])
        
    def call(self,x):
        #x = tf.expand_dims(x,0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        #x = self.conv4(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
    
#     def make(self, input_shape=(227,227,1)):
#         
#         x = tf.keras.layers.Input(shape=input_shape)
#         model = tf.keras.Model(inputs=[x], outputs=self.call(x), name='actor')
#         print(model.summary())
#         return model

def data_work():

    train_data = np.load('data_train.npy') #add the training npy file path
    train_data = train_data.T
    t_label = np.load('t_train_corrected.npy') # add the training label npy file path
    bad_labels = np.where(t_label == -1)
    t_correct = np.delete(t_label,bad_labels[0])
    t_correct = t_correct.astype(np.int32)
    num_class = len(set(t_correct))
    t_final = utils.to_categorical(t_correct,num_classes=num_class)
    train_data_correct = np.delete(train_data,bad_labels[0],axis = 0)
    num_samp,dim = np.shape(train_data_correct)
    dim_2 = int(np.sqrt(dim))
    train_data_correct = train_data_correct.reshape(num_samp,dim_2,dim_2)
    data_final = train_data_correct.reshape(train_data_correct.shape+(1,))
    data_final = tf.image.resize_with_pad(data_final,227,227)

    return data_final,t_final

def self_plot(train,test,l1,l2,sav):
    fig = plt.figure(figsize=(18,9))
    ax1 = fig.add_subplot(1,1,1)
    ax1.plot(train,label = l1)
    ax1.plot(test,label = l2)
    ax1.legend()
    plt.savefig(sav+'.png')


def run():

    ti = str(time.ctime()).split()
    fol = 'model_4_' + str.join('_',ti)
    os.mkdir(fol)
    fil = 'model_4_' + str.join('_',ti)
    pth = os.path.join(fol,fil)

    dat,tar = data_work()

    datagen = ImageDataGenerator(rotation_range = 45, 
                            width_shift_range = 0.2,
                            height_shift_range = 0.2,
                            shear_range = 0.2,
                            zoom_range = 0.2,
                            horizontal_flip = True,
                            fill_mode = 'nearest',
                            validation_split = 0.2,
                            featurewise_center= True,
                            featurewise_std_normalization= True)

    datagen.fit(dat)
    np.save(fol+'/data_mean',datagen.mean)
    np.save(fol+'/data_std',datagen.std)

    #gg_mean, gg_std = datagen.mean,datagen.std
    
    train_generator = datagen.flow(dat,tar,batch_size=32,
                                    subset='training')
    
    val_generator = datagen.flow(dat,tar,batch_size=8,subset='validation')

    '''call_back = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                 min_delta=0,
                                                 patience=20,
                                                 verbose=0,
                                                 mode='auto',
                                                 baseline=None,
                                                 restore_best_weights=True)'''

    call_back_2 = tf.keras.callbacks.ModelCheckpoint(filepath=pth,
                                                    monitor='val_accuracy',
                                                    mode='max',
                                                    save_weights_only= True)


    n_epoch = 250
    model = D2CNN()

    model.compile(optimizer = 'adam',
                loss = tf.keras.losses.CategoricalCrossentropy(),
                metrics = ['accuracy'])

    history = model.fit_generator(train_generator,
                        steps_per_epoch= (0.8*len(dat))//32,
                        validation_data = val_generator,
                        validation_steps = 8,
                        epochs = n_epoch,
                        callbacks = [call_back_2])

    self_plot(history.history['loss'],history.history['val_loss'],'training loss','testing_loss',fol+'/loss')
    self_plot(history.history['accuracy'],history.history['val_accuracy'],'training accuracy','testing_accuracy',fol+'/accuracy')

    model.save_weights(pth)
    model.save(f'model_4.md')

if __name__ == '__main__':
    run()

    
    
    

