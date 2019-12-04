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


start_time = time.time()

image_resize = 60
nwigth = image_resize
nheight = image_resize


#%%
#We select the labels from the N main breeds
labels = labels_filtered["breed"].as_matrix()
labels = labels.reshape(labels.shape[0],1) #labels.shape[0] looks faster than using len(labels)
labels.shape

labels_name, labels_bin = funcoes.matrix_Bin(labels = labels)
labels_bin[0:9]

#%%
for breed in range(len(labels_name)):
    print('Breed {0} : {1}'.format(breed,labels_name[breed]))

#%%
#You can proceed backward with np.argmax to find the breed of an image
labels_cls = np.argmax(labels_bin, axis=1)
labels_name[labels_cls[2]]

i=11
print(labels_name[labels_cls[i]])
lum_img = train_filtered[i,:,:,:]
plt.imshow(lum_img)
plt.show()


#%%
num_validation = 0.30
X_train, X_validation, y_train, y_validation = train_test_split(train_filtered, labels_bin, test_size=num_validation, random_state=6)
X_train.shape
#%%
#from the one-hot database we can find the breed.
df_validation_toPred_cls = np.argmax(y_validation, axis=1)
df_validation_toPred_cls[0:9]
# %%

i=2
print(labels_name[df_validation_toPred_cls[i]])
print(df_validation_toPred_cls[i])
lum_img = X_validation[i,:,:,:]
plt.imshow(lum_img)
plt.show()


# %%
#%%
# Our images are 100 pixels in each dimension.
img_size = image_resize

# Number of colour channels for the images: 3
num_channels = 3

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

# Image Shape
img_shape = (img_size, img_size, num_channels)

# Number of classes : 5 breeds
num_classes = Nber_of_breeds


#%%
x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels]) #-1 put everything as 1 array
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)
keep_prob_fc=tf.placeholder(tf.float32)
keep_prob_conv=tf.placeholder(tf.float32)

#%% [markdown]
# ### **4. CNN with Tensorflow - Design the layer**
# 
# In this part, you can play with the filter sizes and the number of filters. The best model is ont with the proper number of layers but also a good choice of filter sizes and number of filters.

#%%
# Convolutional Layer 1.
filter_size1 = 5          # Convolution filters are 5 x 5 pixels.
num_filters1 = 32         # There are 32 of these filters.

# Convolutional Layer 2.
filter_size2 = 4          # Convolution filters are 4 x 4 pixels.
num_filters2 = 64      # There are 64 of these filters.

# Convolutional Layer 3.
filter_size3 = 3          # Convolution filters are 3 x 3 pixels.
num_filters3 = 128      # There are 128 of these filters.

# Fully-connected layer.
fc_size = 500 



#%%
layer_conv1, weights_conv1 =     funcoes.new_conv_layer(input=x_image,
                   num_input_channels=num_channels,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True,
                   use_dropout=False , keep_prob_conv=keep_prob_conv)
    
layer_conv2, weights_conv2 =     funcoes.new_conv_layer(input=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=True,
                   use_dropout=False , keep_prob_conv=keep_prob_conv)
    
layer_conv3, weights_conv3 =     funcoes.new_conv_layer(input=layer_conv2,
                   num_input_channels=num_filters2,
                   filter_size=filter_size3,
                   num_filters=num_filters3,
                   use_pooling=True,
                   use_dropout=True, keep_prob_conv=keep_prob_conv)


#%%
layer_flat, num_features = funcoes.flatten_layer(layer_conv3)


#%%
#Train
layer_fc1 = funcoes.new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True,
                         use_dropout=True,keep_prob_fc=keep_prob_fc)

layer_fc1

layer_fc2 = funcoes.new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         use_relu=False,
                         use_dropout=False,keep_prob_fc=keep_prob_fc)

layer_fc2


#%%
#Prediction :
y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, axis=1)

#%% [markdown]
# ### **5. CNN with Tensorflow - Definition of the cost function (cross-entropy)**

#%%
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=y_true)
cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#%% [markdown]
# ### **6. CNN with Tensorflow - Training of the CNN**

#%%
session = tf.Session()

def init_variables():
    session.run(tf.global_variables_initializer())

#%% [markdown]
# The function below create a batch from a dataset. We use batch to train our model.

#%%
batch_size = 50


#%%
init_variables()


def optimize(num_iterations, X):
    total_iterations = 0
    start_time = time.time()
    
    #array to plot
    losses = {'train':[], 'validation':[]}
    
    for i in range(num_iterations):
            total_iterations += 1
            # Get a batch of training examples.
            # x_batch now holds a batch of images and
            # y_true_batch are the true labels for those images.
            x_batch, y_true_batch = funcoes.next_batch(batch_size, X_train, y_train)

            # Put the batch into a dict with the proper names
            # for placeholder variables in the TensorFlow graph.
            feed_dict_train = {x: x_batch,
                               y_true: y_true_batch,
                               keep_prob_conv : 0.3,
                               keep_prob_fc : 0.4}
            feed_dict_validation = {x: X_validation,
                               y_true: y_validation,
                               keep_prob_conv : 1,
                               keep_prob_fc : 1}
            
            # Run the optimizer using this batch of training data.
            # TensorFlow assigns the variables in feed_dict_train
            # to the placeholder variables and then runs the optimizer.
            session.run(optimizer, feed_dict=feed_dict_train)
            
            acc_train = session.run(accuracy, feed_dict=feed_dict_train)
            acc_validation = session.run(accuracy, feed_dict=feed_dict_validation)
            losses['train'].append(acc_train)
            losses['validation'].append(acc_validation)
            
            # Print status every X iterations.
            if (total_iterations % X == 0) or (i ==(num_iterations -1)):
            # Calculate the accuracy on the training-set.
                                
                msg = "Iteration: {0:>6}, Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%}"
                print(msg.format(total_iterations, acc_train, acc_validation))
                
    
    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
    
    plt.plot(losses['train'], label='Training loss')
    plt.plot(losses['validation'], label='Validation loss')
    plt.legend()
    _ = plt.ylim()


#In[]
# Aqui treina a rede
optimize(num_iterations=1000, X=5)


#%%
feed_dict_validation = {x: X_validation,
                    y_true: y_validation,
                    keep_prob_conv : 1,
                    keep_prob_fc : 1}
df_validation_Predicted_cls = session.run(y_pred_cls, feed_dict=feed_dict_validation)




#%%
i = 63
print(("True : {0} / {1}").format(df_validation_toPred_cls[i], labels_name[df_validation_toPred_cls[i]]))
print(("Pred : {0} / {1}").format(df_validation_Predicted_cls[i], labels_name[df_validation_Predicted_cls[i]]))

lum = X_validation[i,:,:,:]
plt.imshow(lum)
plt.show()
