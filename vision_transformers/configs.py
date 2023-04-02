import h5py
import math
import torch
import vit_pytorch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from livelossplot import PlotLosses
from sklearn.utils import shuffle
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc
from torch.utils.data import DataLoader, TensorDataset

batch_size = 64
epochs = 50
learning_rate = 1e-3
electron_dir = '../common_task_1/data/SingleElectronPt50_IMGCROPS_n249k_RHv1.hdf5'
photon_dir = '../common_task_1/data/SinglePhotonPt50_IMGCROPS_n249k_RHv1.hdf5'
weights_dir = './weights'
logs_dir = './logs'

def get_features_and_labels(file_dir = None):  
  file = h5py.File(file_dir, 'r')
  return np.array(file['X']), np.array(file['y'])

def get_torch_data_loader(features = None, labels = None, batch_size = batch_size, shuffle = True):
  dataset = TensorDataset(torch.from_numpy(features).permute(0,3,1,2), torch.from_numpy(labels))
  return DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)