# We upload all the packages we need
# In[]
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


# %%
