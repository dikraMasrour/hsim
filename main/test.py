from hsim.models.resnet_mlbs import RESNET_BINARY_MLBS
from hsim.models.baseline import BINARY_BASELINE
from hsim.models.mlbs import BINARY_MLBS

import tensorflow as tf

import argparse
import numpy as np
import h5py 


def get_test_samples(path):
    xtest = np.load(path + 'Xtest.npy')
    ytest = np.load(path + 'ytest.npy')
    return xtest, ytest

def get_model(path):
    if 'baseline' in path:
        engine = BINARY_BASELINE(200, 2)
        engine.build((1, 200, 1))
        engine.load_weights(path + 'baseline_binary_weights.h5')
        engine.compile(loss = 'binary_crossentropy',
                                optimizer = tf.keras.optimizers.Adam(),
                                metrics = tf.keras.metrics.BinaryAccuracy())
        return engine
    
    elif 'mlbs' in path:
        if '15' in path: 
            engine = BINARY_MLBS(200, 2, band_selection_ratio=0.15, filtering_slope=5, sampler_slope=10)
            engine.build((1, 1, 200))
            engine.load_weights(path + 'binary_mlbs_weights_0.15.h5')
        elif '30' in path: 
            engine = BINARY_MLBS(200, 2, band_selection_ratio=0.3, filtering_slope=5, sampler_slope=10)
            engine.build((1, 1, 200))
            engine.load_weights(path + 'binary_mlbs_weights_0.3.h5')
        elif '50' in path: 
            engine = BINARY_MLBS(200, 2, band_selection_ratio=0.5, filtering_slope=5, sampler_slope=10)
            engine.build((1, 1, 200))
            engine.load_weights(path + 'binary_mlbs_weights_0.5.h5')
        elif '75' in path: 
            engine = BINARY_MLBS(200, 2, band_selection_ratio=0.75, filtering_slope=5, sampler_slope=10)
            engine.build((1, 1, 200))
            engine.load_weights(path + 'binary_mlbs_weights_0.75.h5')

        engine.compile(loss='binary_crossentropy',
                                optimizer = tf.keras.optimizers.Adam(),
                                metrics = tf.keras.metrics.BinaryAccuracy())
        return engine



# parser = argparse.ArgumentParser(description="Energy consumption measurement",
#                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument("-model", "--model", action="store", choices=['baseline_binary', 'mlbs_binary'], help="model type", required=True)
# parser.add_argument("-r", "--band selec ratio", action="store", choices=[0.15, 0.3, 0.5, 0.75], type=float, help="band selection ratio")

# args = parser.parse_args()
# config = vars(args)
# if 'baseline' in config['model'] and config['band selec ratio']:
#     print('ERROR: baseline model does not have argument named band selection ratio')

# base_path = 'hsim/logs/'
# model = config['model']
# band_selection_ratio = config['band selec ratio']

# if 'baseline' in model:
#     path = base_path + 'cnn_baseline/'
# else:
#     base_path = base_path + 'mlbs_'
#     if band_selection_ratio==0.15: path = base_path + '15/'
#     elif band_selection_ratio==0.3: path = base_path + '30/'
#     elif band_selection_ratio==0.5: path = base_path + '50/'
#     elif band_selection_ratio==0.75: path = base_path + '75/'


# xtest, ytest = get_test_samples(path)
# engine = get_model(path)
# engine.model().summary()

# print('Number of Test Samples:', xtest.shape[0])
# scores = engine.evaluate(xtest, ytest)

# print('\nBinary Accuracy', scores[1])
# print('Loss', scores[0])
    

from hsim.datasets import load_data
 
# x, y = load_data('indian_pines')
# flat_y = y.flatten()
# corr0 = np.load('hsim\logs\cnn_baseline\corrected_index0.npy')
# corr00 = np.load('hsim\logs\cnn_baseline\corrected_index00.npy')
# corr7 = np.load('hsim\logs\cnn_baseline\corrected_index7.npy')
# testin = np.load('hsim\logs\cnn_baseline\\test_index.npy')
# trainin = np.load('hsim\logs\cnn_baseline\\train_index.npy')
# ytest = np.load('hsim\logs\cnn_baseline\ytest.npy')

# # print(trainin, trainin.shape)
# print(testin, testin.shape)
# corrected = list(sorted(np.concatenate((trainin, testin))))
# corr0 = sorted(corr0)
# corr00 = sorted(corr00)
# corr7 = sorted(corr7)

corrected = [4, 5, 6]
corr7 = [3,5,6]

for i in corr7:
    print(i)
    if i in corrected:
        corrected.insert(i, i)
        corrected[i+1:] = [x+1 for x in corrected[i+1:]] 
        print(corrected)
    else:
        corrected.insert(i)
        print(corrected)

# for i in corr00:
#     if i in corrected:
#         corrected.insert(i, i)
#         corrected[i+1:] = [x+1 for x in corrected[i+1:]] 
#     else:
#         corrected.append(i)

# for i in corr0:
#     if i in corrected:
#         corrected.insert(i, i)
#         corrected[i+1:] = [x+1 for x in corrected[i+1:]] 
#     else:
#         corrected.append(i)

corrected.sort()
print(corrected)

# for index in corrected:
#     flat_y[corrected[index]] = 0.0

