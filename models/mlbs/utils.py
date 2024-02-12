import numpy as np
import json
import h5py
import csv
import tensorflow as tf


def lr_scheduler(epoch, lr):
  if epoch == 50:
        lr = lr / 10
  if epoch == 75:
        lr = lr / 10
  return lr

def compute_class_weights(X_train, y_train):
  train_samples = X_train.shape[0]  
  num_classes = np.unique(y_train).shape[0]

  # per class frequency
  class_frequency = []
  class_weights = []
  for i in range(num_classes):
      class_frequency.append(np.sum(y_train == i))
      class_weights.append(train_samples/class_frequency[i])
      
  class_weights = np.array(class_weights)
  # normalisation of class weights
  class_weights_norm = class_weights / np.sum(class_weights)
  class_weights_norm = np.sqrt(class_weights_norm)

  class_weights_dic = {}
  for i in range(num_classes):
      class_weights_dic[i] = 100 * class_weights_norm[i]
  return class_weights_dic


def read_history(path='hsim\\models\\binayr_mlbs\\binary_mlbs_history.json'):
    with open(path, mode='r') as f:
        json.load(f)

def save_model_weights(model, weights_path):
    model.save_weights(weights_path)

def load_model_weights(fold_no, path='hsim\models\hsim_models_binary_mlbs_weights.h5'):
    with h5py.File(path, "r") as f:
        group = 'weights_'+ str(fold_no) +'/Variable:0'
        V = f[group][0]
        return V
    
def extract_mask_from_weights(weights, UK, filtering_slope=5, sampler_slope=10, sparse_ratio=0.1500):
    # sigmoid
    S = tf.math.sigmoid(filtering_slope * weights)

    # normalisation (forces sparsity)
    s = tf.reduce_mean(S, axis=0)
    cond = tf.cast(tf.greater_equal(s, sparse_ratio), dtype=tf.float32)
    ge = (sparse_ratio / s) * S
    lt = 1.0 - (((1.0 - sparse_ratio) / (1.0 - s)) * (1.0 - S))
    N = cond * ge + (1 - cond) * lt

    # thresholding operation
    # mask = tf.math.sigmoid(sampler_slope * (N - UK))
    mask = float(N > UK)

    mask_array = mask.numpy()
    return mask_array


def save_evaluation_results(scores, preds, test, fold_no, path):
    with open(str(path)+'\\scores'+str(fold_no)+'.csv','w') as f:
    # with open(str(path)+'scores_'+str(fold_no)+'.csv','w') as f: #*GoogleColab
        w = csv.writer(f)
        w.writerow(['loss', 'accuracy'])
        w.writerows([scores])

    np.save(str(path)+'\\preds_'+str(fold_no)+'.npy', preds)
    np.save(str(path)+'\\test_'+str(fold_no)+'.npy', test)
    # np.save(str(path)+'preds_'+str(fold_no)+'.npy', preds) #*GoogleColab
    # np.save(str(path)+'test_'+str(fold_no)+'.npy', test) #*GoogleColab