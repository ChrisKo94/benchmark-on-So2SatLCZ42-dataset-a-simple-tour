# @Date:   2020-02-04T17:07:00+01:00
# @Last modified time: 2020-04-24T10:31:52+02:00

import resnet
import model
import lr

from dataLoader import generator

import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Nadam
from keras.callbacks import EarlyStopping
# from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
import os

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

###################################################
'path to save models from check points:'
file0='/data/lcz42_votes/benchmark-on-So2SatLCZ42-dataset-a-simple-tour/results/'
#file0 = 'C:/Users/koll_ch/PycharmProjects/benchmark-on-So2SatLCZ42-dataset-a-simple-tour/results/'


'path to data, needs to be set accordingly'
train_file='/data/lcz42_votes/data/train_data.h5'
#train_file = "D:/Data/LCZ_Votes/train_data.h5"
validation_file='/data/lcz42_votes/data/validation_data.h5'
#validation_file = "D:/Data/LCZ_Votes/validation_data.h5"

numClasses=17
batchSize=32
###################################################

'number of all samples in training and validation sets'
trainNumber=158799
validationNumber=30695
lrate = 0.0005
lr_sched = lr.step_decay_schedule(initial_lr=lrate, decay_factor=0.5, step_size=5)

###################################################
# patch_shape=(32,32,10)
# model = resnet.resnet_v2(input_shape=patch_shape, depth=11, num_classes=numClasses)
model = model.sen2LCZ_drop(depth=17, dropRate=0.2, fusion=1)

model.compile(optimizer = Nadam(), loss = 'categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor = 'val_loss', patience = 40)
modelbest = file0 + "Sen2LCZ_" + str(batchSize) + "_lr_ " + str(lrate) +"_weights_best.hdf5"
checkpoint = ModelCheckpoint(modelbest, monitor='val_accuracy', verbose=1, save_best_only=True,
                             save_weights_only=True, mode='auto', save_freq='epoch')

model.fit(generator(train_file, batchSize=batchSize, num=trainNumber),
                steps_per_epoch = trainNumber//batchSize,
                validation_data= generator(validation_file, num=validationNumber, batchSize=batchSize),
                validation_steps = validationNumber//batchSize,
                epochs=100,
                max_queue_size=100,
                callbacks=[early_stopping, checkpoint, lr_sched])
