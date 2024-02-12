import numpy as np
from sklearn import preprocessing
from imblearn.over_sampling import RandomOverSampler, SMOTE
import os


def shuffle_arrays(*args):
    """ shuffle arrays given as arguments the same way along the first axis """
    # check dim
    l = args[0].shape[0]
    for a in args:
        assert a.shape[0] == l
    p = np.random.permutation(args[0].shape[0])
    shuffled_arrays = list()
    for a in args:
        shuffled_arrays.append(a[p])
    return tuple(shuffled_arrays)

def train_test_split(X, y, train_size=0.75):
    """ perform train/test split for each label """
    # shuffle
    X, y = shuffle_arrays(X, y)
    # select train_size data points per label
    X_trains, X_tests = list(), list()
    y_trains, y_tests = list(), list()
    for l in np.unique(y).tolist():
        idx = np.argwhere(y == l).flatten()
        cut = int(train_size * idx.shape[0])
        X_train, X_test = X[idx[:cut]],  X[idx[cut:]]
        y_train, y_test = np.repeat(l, X_train.shape[0]), np.repeat(l, X_test.shape[0])
        X_trains.append(X_train)
        X_tests.append(X_test)
        y_trains.append(y_train)
        y_tests.append(y_test)
    X_train, y_train = np.vstack(X_trains), np.hstack(y_trains)
    X_test, y_test = np.vstack(X_tests), np.hstack(y_tests)
    # shuffle again
    X_train, y_train = shuffle_arrays(X_train, y_train)
    X_test, y_test = shuffle_arrays(X_test, y_test)
    return (X_train, y_train), (X_test, y_test)

def preprocess(X, y, flatten=True, correct=None):
    if flatten:
        X = X.flatten().reshape((-1, X.shape[2]))
        y = y.flatten()
    # correction (remove unwanted label) requires flattening
    if correct is not None:
        # find indices of the class to be removed
        index = [i for i in range(y.shape[0]) if y[i] == correct]
        if os.path.isfile('corrected_index'+str(correct)+'.npy'):
            np.save('corrected_index'+str(correct)+str(correct), np.array(index))
        else:
            np.save('corrected_index'+str(correct), np.array(index))
        # corrected y
        y = np.delete(y, index)
        y = np.where(y < correct, y, y-1)
        # corrected X
        X = np.delete(X, index, axis=0)
    return X, y

def normalise(X):
    # if len(X.shape) == 2:
    #     max_vals = X.max(axis=0, keepdims=True)
    #     min_vals = X.min(axis=0, keepdims=True)
    #     X = (X - min_vals) / (max_vals - min_vals)
    # elif len(X.shape) == 3:   
    #     max_vals = X.max(axis=2, keepdims=True)
    #     min_vals = X.min(axis=2, keepdims=True)
    #     X = (X - min_vals) / (max_vals - min_vals)
    min_val = np.amin(X, axis=(0,1))
    max_val = np.amax(X, axis=(0,1))
    data_norm = (X - min_val) / (max_val - min_val)
    return data_norm

def make_binary_labels(y, labels_to_keep, flattened=True):
    if flattened:
        for i in range(len(y)):
            if np.all(y[i] in labels_to_keep):
                y[i] = 1.
            else:
                y[i] = 0.
        y = np.array(y, dtype=float)
    # else:
    #     y = np.where(y==label_to_keep, 1., y)
    #     y = np.where(y!=1., 0., y)
    return y

def oversample_binary(x, y, strategy='minority'):
    # define oversampling strategy
    oversample = RandomOverSampler(sampling_strategy=strategy)
    # fit and apply the transform
    X_over, y_over = oversample.fit_resample(x, y)
    return X_over, y_over

def oversample_multiclass(x, y):
    # define oversampling strategy
    oversample = SMOTE(k_neighbors=1)
    # fit and apply the transform
    Xf_over, yf_over = oversample.fit_resample(x, y)
    return Xf_over, yf_over