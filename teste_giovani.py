# In[]
#get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import math
import os
import scipy.misc
from scipy.stats import itemfreq
from random import sample
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Image manipulation.
import PIL.Image
from IPython.display import display
#from resizeimage import resizeimage

# Panda
import pandas as pd

# Open a Zip File
from zipfile import ZipFile
from io import BytesIO

import funcoes

# check tensorflow version
tf.__version__

# variavel para facilitar carregar as coisas para o workspace
PREFIXO = './'

# %%
# We unzip the train and test zip file
archive_train = ZipFile(PREFIXO + "/Data/train.zip", 'r')
archive_test = ZipFile(PREFIXO + "/Data/test.zip", 'r')

# %%
# **3. Resize and normalize the data**
print("normalizando os dados e resize")
image_resize = 60
funcoes.DataBase_creator(archivezip = archive_train, nwigth = image_resize, nheight = image_resize , save_name = "train")
funcoes.DataBase_creator(archivezip = archive_test, nwigth = image_resize, nheight = image_resize , save_name = "test")


#%%
#load TRAIN
print('carregando train e test')
train = pickle.load( open( "train.p", "rb" ) )
print('shape train: ' + str(train.shape))
test = pickle.load( open( "test.p", "rb" ) )
print('shape test: ' + str(test.shape))


# %%
#%%
print('load csv')
df_train= pd.read_csv(PREFIXO+'/Data/labels.csv')
df_train.sample(5)
#%%
#######Upload the zip (input data base)########
labels_raw = pd.read_csv(PREFIXO+'/Data/labels.csv', header=0, sep=',', quotechar='"')

#Check 5 random values
labels_raw.sample(5)
# %%
Nber_of_breeds = 8

labels_filtered_index = funcoes.main_breeds(labels_raw = labels_raw, Nber_breeds = Nber_of_breeds, all_breeds='FALSE')
labels_filtered = labels_raw.iloc[labels_filtered_index[0],:]
train_filtered = train[labels_filtered_index[0],:,:,:]

print('- Number of images remaining after selecting the {0} main breeds : {1}'.format(Nber_of_breeds, labels_filtered_index[0].shape))
print('- The shape of train_filtered dataset is : {0}'.format(train_filtered.shape))
#%%
#print(labels_filtered[90])
lum_img = train_filtered[1,:,:,:]
plt.imshow(lum_img)
plt.show()

# %%
#%%
#We select the labels from the N main breeds
labels = labels_filtered["breed"].as_matrix()
labels = labels.reshape(labels.shape[0],1) #labels.shape[0] looks faster than using len(labels)
labels.shape