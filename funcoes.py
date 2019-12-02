#  esse arquivo pode guardar varias funcoes para facilitar
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


def DataBase_creator(archivezip, nwigth, nheight, save_name):
    # We choose the archive (zip file) + the new wigth and height for all the image which will be reshaped

    # Start-time used for printing time-usage below.
    start_time = time.time()

    # nwigth x nheight = number of features because images are nwigth x nheight pixels
    s = (len(archivezip.namelist()[:])-1, nwigth, nheight, 3)
    allImage = np.zeros(s)

    for i in range(1, len(archivezip.namelist()[:])):
        filename = BytesIO(archivezip.read(archivezip.namelist()[i]))
        image = PIL.Image.open(filename)  # open colour image
        image = image.resize((nwigth, nheight))
        image = np.array(image)
        # 255 = max of the value of a pixel
        image = np.clip(image/255.0, 0.0, 1.0)

        allImage[i-1] = image

    # we save the newly created data base
    pickle.dump(allImage, open(save_name + '.p', "wb"))

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
