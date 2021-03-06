# @Date:   2020-02-04T17:07:00+01:00
# @Last modified time: 2020-04-24T10:31:52+02:00

import resnet
import model
import lr
import h5py
import numpy as np
import pandas as pd

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

all_cities = False
distributional = False
dry_run = True

alpha = 0

###################################################
'path to save models from check points:'
if all_cities:
    file0 = '/data/lcz42_votes/benchmark-on-So2SatLCZ42-dataset-a-simple-tour/results/all_cities/'
else:
    file0 = '/data/lcz42_votes/benchmark-on-So2SatLCZ42-dataset-a-simple-tour/results/'
#file0 = 'C:/Users/koll_ch/PycharmProjects/benchmark-on-So2SatLCZ42-dataset-a-simple-tour/results/'

'path to data, needs to be set accordingly'
if all_cities:
    train_file = '/data/lcz42_cities/train_data.h5'
    validation_file = '/data/lcz42_cities/validation_data.h5'
    path_data = "/data/lcz42_cities/"
else:
    train_file = '/data/lcz42_votes/data/train_data.h5'
    validation_file = '/data/lcz42_votes/data/validation_data.h5'
    path_data = "/data/lcz42_votes/data/"

numClasses=17
batchSize=64
###################################################

#mode = "all"
mode = "urban"
uncertain = False
entropy_quantile = 0# choose quantile of most certain images (w.r.t. voter entropy) for training, requires mode = "urban"

train_data = h5py.File(train_file, 'r')
x_train = np.array(train_data.get("x"))
y_train = np.array(train_data.get("y"))

validation_data = h5py.File(validation_file, 'r')
x_val = np.array(validation_data.get("x"))
y_val = np.array(validation_data.get("y"))

if mode == "urban":
    indices_train = np.where(np.where(y_train == np.amax(y_train, 0))[1] + 1 < 11)[0]
    x_train = x_train[indices_train, :, :, :]
    indices_val = np.where(np.where(y_val == np.amax(y_val, 0))[1] + 1 < 11)[0]
    x_val = x_val[indices_val, :, :, :]

    if not distributional:
        y_train = y_train[indices_train]
        y_val = y_val[indices_val]

if entropy_quantile > 0 and mode == "urban":
    entropies = h5py.File(path_data + "entropies_train.h5", 'r')
    entropies_train = np.array(entropies.get("entropies_train"))
    entropies_train = entropies_train[indices_train]
    entropies_train[np.where(np.isnan(entropies_train))] = 0

    entropies = pd.DataFrame({"entropies": entropies_train,
                              "order": np.arange(len(y_train))})
    if uncertain == False:
        entropies = entropies.sort_values(by=['entropies'])
    else:
        entropies = entropies.sort_values(by=['entropies'], ascending=False)
    ## Order training data accordingly
    idx = np.array(entropies["order"])
    ## Cut off at given quantile
    idx = idx[:np.floor(entropy_quantile * len(idx)).astype(int)]
    x_train = x_train[idx, :, :, :]
    y_train = y_train[idx]

if distributional:
    if alpha > 0:
        train_distributions = h5py.File('/data/lcz42_votes/data/train_label_distributions_data' + '_alpha_' + str(alpha) + '.h5', 'r')
        y_train = np.array(train_distributions['train_label_distributions'])
        val_distributions = h5py.File('/data/lcz42_votes/data/val_label_distributions_data' + '_alpha_' + str(alpha) + '.h5', 'r')
        y_val = np.array(val_distributions['val_label_distributions'])
    else:
        train_distributions = h5py.File('/data/lcz42_votes/data/train_label_distributions_data.h5', 'r')
        y_train = np.array(train_distributions['train_label_distributions'])
        val_distributions = h5py.File('/data/lcz42_votes/data/val_label_distributions_data.h5', 'r')
        y_val = np.array(val_distributions['val_label_distributions'])



'number of all samples in training and validation sets'
trainNumber=y_train.shape[0]
validationNumber=y_val.shape[0]

lrate = 0.002
lr_sched = lr.step_decay_schedule(initial_lr=lrate, decay_factor=0.5, step_size=5)

###################################################
# patch_shape=(32,32,10)
# model = resnet.resnet_v2(input_shape=patch_shape, depth=11, num_classes=numClasses)
if mode == "urban":
    model = model.sen2LCZ_drop(depth=17, dropRate=0.2, fusion=1, num_classes=10)
else:
    model = model.sen2LCZ_drop(depth=17, dropRate=0.2, fusion=1)

if distributional:
    model.compile(optimizer=Nadam(), loss='KLDivergence', metrics=['KLDivergence'])
else:
    model.compile(optimizer=Nadam(), loss='categorical_crossentropy', metrics=['accuracy'])

if dry_run:
    early_stopping = EarlyStopping(monitor = 'val_loss', patience = 100)
else:
    early_stopping = EarlyStopping(monitor='val_loss', patience=40)

PATH = file0 + "Sen2LCZ_" + str(batchSize) + "_lr_" + str(lrate)
if mode == "urban":
    PATH = PATH + "_urban"

if distributional:
    if alpha > 0:
        PATH = PATH + "_d" + "alpha_" + str(alpha)
    else:
        PATH = PATH + "_d"

if entropy_quantile > 0:
    if uncertain == True:
        PATH = PATH + "_most_uncertain_" + str(entropy_quantile)
    else:
        PATH = PATH + "_most_certain_" + str(entropy_quantile)

if dry_run:
    PATH = PATH + "_dry_run"

modelbest = PATH + "_weights_best.hdf5"

if distributional:
    checkpoint = ModelCheckpoint(modelbest, monitor='val_kullback_leibler_divergence', verbose=1, save_best_only=True,
                                 save_weights_only=True, mode='auto', save_freq='epoch')
else:
    checkpoint = ModelCheckpoint(modelbest, monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=True, mode='auto', save_freq='epoch')

model.fit(generator(x_train, y_train, batchSize=batchSize, num=trainNumber, mode=mode),
                steps_per_epoch = trainNumber//batchSize,
                validation_data= generator(x_val, y_val, num=validationNumber, batchSize=batchSize, mode=mode),
                validation_steps = validationNumber//batchSize,
                epochs=300,
                max_queue_size=100,
                callbacks=[early_stopping, checkpoint, lr_sched])
