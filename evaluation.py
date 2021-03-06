# @Date:   2020-04-24T10:19:05+02:00
# @Last modified time: 2020-04-25T11:59:59+02:00

import model

import numpy as np
import h5py
import scipy.io as scio
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix

import tensorflow as tf
import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.45
#session = tf.Session(config=config)

all_cities = False
distributional = False

alpha = 0.08

def predata4LCZ(file, keyX, keyY):
    hf = h5py.File(file, 'r')
    x_tra = np.array(hf[keyX])
    y_tra = np.array(hf[keyY])
    hf.close()

    print(x_tra.shape, y_tra.shape)

    return x_tra, y_tra
################################################################################

if all_cities:
    file0 = 'C:/Users/koll_ch/PycharmProjects/benchmark-on-So2SatLCZ42-dataset-a-simple-tour/results/all_cities/'
else:
    file0 = 'C:/Users/koll_ch/PycharmProjects/benchmark-on-So2SatLCZ42-dataset-a-simple-tour/results/'

mode = "urban"
#mode = all

if mode == "urban":
    model = model.sen2LCZ_drop(depth=17, num_classes=10, dropRate=0.2, fusion=1)
else:
    model = model.sen2LCZ_drop(depth=17, dropRate=0.2, fusion=1)

batch_size = 64#8 16 32
lrate = 0.002
numC= 17 ;

'loading test data'
#file='/data/lcz42_votes/data/test_data.h5'
#file='D:/Data/LCZ_Votes/test_data.h5'

if all_cities:
    file='D:/Data/LCZ42_Cities/test_data.h5'
else:
    file = 'D:/Data/LCZ_Votes/test_data.h5'

x_tst, y_tst= predata4LCZ(file, 'x', 'y')

if mode == "urban":
    indices_test = np.where(np.where(y_tst == np.amax(y_tst, 0))[1] + 1 < 11)[0]
    x_tst = x_tst[indices_test, :, :, :]
    y_tst = y_tst[indices_test, :10]

patch_shape = (32, 32, 10)


#########################################
if mode == "urban":
    if distributional:
        if alpha > 0:
            modelbest = file0  + "Sen2LCZ_" + str(batch_size) + "_lr_" + str(lrate) + "_urban_d" + str(alpha) + "_weights_best.hdf5"
        else:
            modelbest = file0 + "Sen2LCZ_" + str(batch_size) + "_lr_" + str(lrate) + "_urban_d" + "_weights_best.hdf5"
    else:
        modelbest = file0 + "Sen2LCZ_" + str(batch_size) + "_lr_" + str(lrate) + "_urban" + "_weights_best.hdf5"
else:
    modelbest = file0 + "Sen2LCZ_" + str(batch_size) + "_lr_" + str(lrate) + "_weights_best.hdf5"

'load saved best model'
model.load_weights(modelbest, by_name=False)

# 4. test phase
y_pre = model.predict(x_tst, batch_size = batch_size)
y_pre = y_pre.argmax(axis=-1)+1
y_testV = y_tst.argmax(axis=-1)+1

# Add training data as well
#y_pre_trn = model.predict(x_trn, batch_size = batch_size)
#y_pre_trn = y_pre_trn.argmax(axis=-1)+1
#y_trnV = y_trn.argmax(axis=-1)+1

#y_pre = np.hstack((y_pre, y_pre_trn))
#y_testV = np.hstack((y_testV, y_trnV))

labels=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "A", "B", "C", "D", "E", "F", "G"]

if mode == "urban":
    labels = labels[:10]

C = confusion_matrix(y_testV-1, y_pre-1, normalize="true")
conf_mat = pd.DataFrame(C, index=labels, columns=labels)
sns.heatmap(conf_mat * 100, annot=True, fmt = ".0f", cmap="summer")
plt.show()
# print(type(C))

classRep = classification_report(y_testV, y_pre, digits = 4)
oa = accuracy_score(y_testV, y_pre)
cohKappa = cohen_kappa_score(y_testV, y_pre)

print('#####################classwise accuracy:')
print(classRep)

print('#####################overall accuracy and Kappa:')
print(oa, cohKappa)

#scio.savemat((file0 + 'Acc' + "_" + str(batch_size)+'.mat'), {'classRep': classRep ,'oa': oa, 'cohKappa': cohKappa, 'confusion_matrix': np.int64(C), 'y_testV':y_testV, 'y_pre':y_pre})
#
