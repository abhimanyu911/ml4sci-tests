import h5py
import math
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from livelossplot import PlotLosses
from torch.optim.lr_scheduler import LambdaLR
from .configs import *
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

def get_features_and_labels(file_dir = None):  
  file = h5py.File(file_dir, 'r')
  return np.array(file['X']), np.array(file['y'])

def get_torch_data_loader(features = None, labels = None, batch_size = batch_size, shuffle = True):
  dataset = TensorDataset(torch.from_numpy(features).permute(0,3,1,2), torch.from_numpy(labels))
  return DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)
    
def get_tflow_model():
    
  model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(96, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(1, activation='sigmoid')
  ])

  return model

class ParticleClassifier(nn.Module):
  def __init__(self):
    
    super(ParticleClassifier, self).__init__()
    self.conv1 = nn.Conv2d(2, 16, 3)
    self.conv2 = nn.Conv2d(16, 16, 3)
    self.conv3 = nn.Conv2d(16, 32, 3)
    self.conv4 = nn.Conv2d(32, 32, 3)
    self.pool = nn.MaxPool2d(2, 2)
    self.dropout = nn.Dropout(0.25)
    self.fc1 = nn.Linear(32 * 2 * 2, 96)
    self.fc2 = nn.Linear(96, 16)
    self.fc3 = nn.Linear(16, 1)

  def forward(self, x):

    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = self.pool(x)
    x = F.relu(self.conv3(x))
    x = self.pool(x)
    x = F.relu(self.conv4(x))
    x = self.pool(x)
    x = self.dropout(x)
    x = x.reshape(-1, 32 * 2* 2)
    x = F.relu(self.fc1(x))
    x = self.dropout(x)
    x = F.relu(self.fc2(x))
    x = self.dropout(x)
    x = self.fc3(x)
    return F.sigmoid(x)
  
class ParticleClassifierLightning(pl.LightningModule):
  def __init__(self):
    super(ParticleClassifierLightning, self).__init__()
    self.model = ParticleClassifier()
    self.loss = nn.BCELoss()

  def forward(self, x):
    return self.model(x)

  def training_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self(x).squeeze()
    loss = self.loss(y_hat, y)
    auc = roc_auc_score(y.detach().cpu().numpy(), y_hat.detach().cpu().numpy())
    self.log("train_loss", loss, on_step = False, on_epoch = True, prog_bar = True)
    self.log("train_auc", auc, on_step = False, on_epoch = True, prog_bar = True)
    return loss

  def validation_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self(x).squeeze()
    loss = self.loss(y_hat, y)
    auc = roc_auc_score(y.detach().cpu().numpy(), y_hat.detach().cpu().numpy())
    self.log("val_loss", loss, on_step = False, on_epoch = True, prog_bar = True)
    self.log("val_auc", auc, on_step = False, on_epoch = True, prog_bar = True)
    return loss

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda, verbose=True)
    return [optimizer], [scheduler]
  
def scheduler(epoch, lr):
  if epoch < 10:
    return lr
  else:
    return lr * tf.math.exp(-0.1)
  
def lr_lambda(epoch):
  if epoch <= 9:
    return 1
  else:
    return math.exp(-0.1 * (epoch - 9))

def plot_ROC_AUC(y_true = None, y_pred = None):
  
  fpr, tpr, _ = roc_curve(y_true, y_pred)
  roc_auc = auc(fpr, tpr)
  plt.figure()
  plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.4f)' % roc_auc)
  plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label = 'Random Guessing (AUC = 0.5)')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.0])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver operating characteristic')
  plt.legend(loc="lower right")
  plt.show()

def plot_learning_curves(file_path):

  # Read the CSV file into a pandas DataFrame
  df = pd.read_csv(file_path, header=0)

  # Preprocess the CSV file
  epochs = np.unique(df['epoch'].dropna().values)
  val_loss = df['val_loss'].dropna().values
  val_auc = df['val_auc'].dropna().values
  train_loss = df['train_loss'].dropna().values
  train_auc = df['train_auc'].dropna().values

  # Create the side-by-side plots
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

  # Plot train and validation AUC vs epochs
  ax1.plot(epochs, train_auc, label='Train AUC')
  ax1.plot(epochs, val_auc, label='Validation AUC')
  ax1.set_xlabel('Epochs')
  ax1.set_ylabel('AUC')
  ax1.set_title('Train and Validation AUC vs Epochs')
  ax1.legend()

  # Plot train and validation loss vs epochs
  ax2.plot(epochs, train_loss, label='Train Loss')
  ax2.plot(epochs, val_loss, label='Validation Loss')
  ax2.set_xlabel('Epochs')
  ax2.set_ylabel('Loss')
  ax2.set_title('Train and Validation Loss vs Epochs')
  ax2.legend()

  plt.show()
