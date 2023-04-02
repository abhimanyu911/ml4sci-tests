from random import shuffle
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
from math import ceil
import tensorflow as tf
import matplotlib.pyplot as plt
import os
data_dir = './data'
train_dir = data_dir + '/train'
val_dir = data_dir + '/val'
test_dir = data_dir + '/test'
weights_dir = './weights/best_weights'
val_samples = 3483
test_samples = 3483
train_samples = 139306 - val_samples - test_samples
batch_size = 64
epochs = 10
learning_rate = 1e-3

def HDF5ImageGenerator(hdf5_x = None, hdf5_y = None, batch_size = None, mode = 'train'):
    sample_count  = hdf5_x.shape[0]

    while True:
        batch_index = 0
        batches_list = list(range(int(ceil(float(sample_count) / batch_size))))
        if mode == 'train':
            shuffle(batches_list)

        while batch_index < len(batches_list):
            batch_number = batches_list[batch_index]
            start        = batch_number * batch_size
            end          = min(start + batch_size, sample_count)

            # Load data from disk
            x = hdf5_x[start: end]
            y = hdf5_y[start: end]

            # Augment batch

            batch_index += 1
            

            yield x,y

def scheduler(epoch, lr):
  if epoch < 10:
    return lr
  else:
    return lr * tf.math.exp(-0.1)
  

def center_crop(image, new_height, new_width):
  height, width = image.shape[:2]
  start_row = (height - new_height) // 2
  start_col = (width - new_width) // 2
  cropped_image = image[start_row:start_row+new_height, start_col:start_col+new_width]
  return cropped_image

def plot_ROC_AUC(y_true = None, y_pred = None):
  
  fpr, tpr, _ = roc_curve(y_true, y_pred)
  roc_auc = auc(fpr, tpr)
  plt.figure()
  plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
  plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label = 'Random Guessing (AUC = 0.5)')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.0])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver operating characteristic')
  plt.legend(loc="lower right")
  plt.show()