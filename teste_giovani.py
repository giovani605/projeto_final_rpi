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

# check tensorflow version
tf.__version__

# variavel para facilitar carregar as coisas para o workspace
PREFIXO = './'

# %%
# We unzip the train and test zip file
archive_train = ZipFile("./projeto_final_rpi/Data/train.zip", 'r')
archive_test = ZipFile("./projeto_final_rpi/Data/test.zip", 'r')

# This line shows the 5 first image name of the train database
archive_train.namelist()[0:5]

# This line shows the number of images in the train database
len(archive_train.namelist()[:])-1  # we must remove the 1st value
