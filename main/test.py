from models.resnet_mlbs import RESNET_BINARY_MLBS
from models.baseline import BINARY_BASELINE
from models.mlbs import BINARY_MLBS16, BINARY_MLBS32

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
            engine = BINARY_MLBS32(200, 2, band_selection_ratio=0.15, filtering_slope=5, sampler_slope=10)
            engine.build((1, 1, 200))
            engine.load_weights(path + 'binary_mlbs_weights_0.15.h5')
        elif '30' in path: 
            engine = BINARY_MLBS32(200, 2, band_selection_ratio=0.3, filtering_slope=5, sampler_slope=10)
            engine.build((1, 1, 200))
            engine.load_weights(path + 'binary_mlbs_weights_0.3.h5')
        elif '50' in path: 
            engine = BINARY_MLBS16(200, 2, band_selection_ratio=0.5, filtering_slope=5, sampler_slope=10)
            engine.build((1, 1, 200))
            engine.load_weights(path + 'binary_mlbs_weights_0.5.h5')
        elif '75' in path: 
            engine = BINARY_MLBS16(200, 2, band_selection_ratio=0.75, filtering_slope=5, sampler_slope=10)
            engine.build((1, 1, 200))
            engine.load_weights(path + 'binary_mlbs_weights_0.75.h5')

        engine.compile(loss='binary_crossentropy',
                                optimizer = tf.keras.optimizers.Adam(),
                                metrics = tf.keras.metrics.BinaryAccuracy())
        return engine



parser = argparse.ArgumentParser(description="Energy consumption measurement",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-model", "--model", action="store", choices=['baseline_binary', 'mlbs_binary'], help="model type", required=True)
parser.add_argument("-r", "--band selec ratio", action="store", choices=[0.15, 0.3, 0.5, 0.75], type=float, help="band selection ratio")

args = parser.parse_args()
config = vars(args)
if 'baseline' in config['model'] and config['band selec ratio']:
    print('ERROR: baseline model does not have argument named band selection ratio')

base_path = 'logs/'
model = config['model']
band_selection_ratio = config['band selec ratio']

if 'baseline' in model:
    path = base_path + 'cnn_baseline/'
else:
    base_path = base_path + 'mlbs_'
    if band_selection_ratio==0.15: path = base_path + '15/'
    elif band_selection_ratio==0.3: path = base_path + '30/'
    elif band_selection_ratio==0.5: path = base_path + '50/'
    elif band_selection_ratio==0.75: path = base_path + '75/'


xtest, ytest = get_test_samples(path)
engine = get_model(path)
engine.model().summary()

print('Number of Test Samples:', xtest.shape[0])
preds = engine.predict(xtest)

scores = engine.evaluate(xtest, ytest)
print('\nBinary Accuracy', scores[1])
print('Loss', scores[0])

# Prediction for single sample
# sample = 0
# pred = engine.predict(np.array([xtest[sample]]))
# score = engine.evaluate(np.array([xtest[sample]]), np.array([ytest[sample]]))
# print(pred)
