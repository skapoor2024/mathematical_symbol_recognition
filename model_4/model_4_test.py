import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

from tensorflow.keras import Model, Sequential, layers, utils
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Activation,MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import seaborn as sns

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

def data_work(data_set,label_set,mean_file,std_file):

    train_data = np.load(data_set)
    train_data = train_data.T
    t_label = np.load(label_set)
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

    mean_load = np.load(mean_file,allow_pickle = True)
    std_load = np.load(std_file,allow_pickle = True)

    data_final = data_final - mean_load
    data_final = data_final / std_load

    return data_final,t_correct,t_final,num_class

def self_plot(cm,sav):

    labels_names_2 = ['x','sqrt','+','-',
                '=','%','d','prod','pi','sum']
    ax = sns.heatmap(cm, annot=True, cmap='Blues',fmt='.0f',annot_kws={'fontsize':14})
    ax.xaxis.set_ticklabels(labels_names_2)
    ax.yaxis.set_ticklabels(labels_names_2)
    fig = ax.get_figure()
    fig.savefig(sav+'.png',dpi=400)

def run():

    data_set = 'data_train.npy' #the dataset to be tested
    label_set = 't_train_corrected.npy' # the correct labels of the dataset
    mean_file = 'data_mean.npy' # the mean of the training data
    std_file = 'data_std.npy' # the std of the training data
    model_folder = 'model_4.md'

    dat,tar,tar_1hot,num_class = data_work(data_set,label_set,mean_file,std_file)
    model = tf.keras.models.load_model(model_folder)
    y_pred = model.predict(dat)
    y_pred_arr = tf.math.argmax(y_pred,axis = 1)
    confusion_matrix = tf.math.confusion_matrix(tar,y_pred_arr,num_class)
    print(confusion_matrix)
    acc = tf.keras.metrics.CategoricalAccuracy()
    acc.update_state(tar_1hot,y_pred)
    print('The accuracy of the model is')
    print(acc.result().numpy())
    print('The predicted ouput is')
    print(y_pred_arr)
    self_plot(confusion_matrix,'cm_data')
    

if __name__ == '__main__':
    run()
    






