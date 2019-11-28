# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), '../../../../tmp'))
	print(os.getcwd())
except:
	pass
#%%
from IPython import get_ipython

#%% [markdown]
# # UPDATE : I perform many updates to provide a better explanation of my code. In addition, I added layers and dropout in my CNN to have better results.
# 
# # Convolutional Neural Network with tensorflow
# 
# My goal is to use Tensorflow (not Keras) for this competition.
# Just a quick word about Keras. In order to have better result, it's better to use pre-trained model which can be export from Keras and do transfert leraning.To quickly summarize, you can reuse some weights from the pre-trained models and add or remove some of the layers in order to create a new model.
# Here we will not do it. We just want t create from scratch our own CNN.
# 
# I share a Python code which shows how to use Tensorflow to build a simple convolutional network (2 layers of convolutional network). I learn a lot from [Hvass-Labs github](https://github.com/Hvass-Labs/TensorFlow-Tutorials) so go to see his tutorial, it's definitely one of the best. As you will see, I copy a lot from his code for the convolutional network part.
# 
# To simplify the game, I will reduce the dataset with the 8 main breeds (take only 3 or 5 if you want to run the code faster)..
# 
# Please let me know if you manage to improve the results (by using Tensorflow) and how did you do? I really need your help as you will see the results are not good. As well, if you find some mistakes or have some questions, do not hesitate to put a comment. I will continue to add more informations on this code to make it more clear.
# 
# ## Architecture
# We will create a Convolutional Neural Network (CNN) which willl be able to classify dogs depending on their breed. Our CNN architecture will be as followed:
# 
# - Convolutional Layer n°1 with 32 filters
#  + Max pooling
#  + Relu
# - Convolutional Layer n°2 with 64 filters
#  + Max pooling
#  + Relu
# - Convolutional Layer n°3 with 128 filters
#  + Max pooling
#  + Relu
#  + DropOut
# - Flatten Layer
# - Fully Connected Layer with 500 nodes
#  + Relu
#  + DropOut
# - Fully Connected Layer with n nodes (n = number of breeds)
# 
# ## Preliminary work
# 
# ### **1. Packages**
# Let's import all the packages we need.

#%%
#We upload all the packages we need
get_ipython().magic('matplotlib inline')
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

#Panda
import pandas as pd

#Open a Zip File
from zipfile import ZipFile
from io import BytesIO

#check tensorflow version
tf.__version__

#%% [markdown]
# As you can see, I'm currently using the version 1.0.1 but you can use any other one less updated like 1.1.0. It should also work.
# 
# ### **2. Unzip the files**
# 
# ## **The following code is design to be run on *your own JUPYTER NOTEBOOK made on your computer*. It will not work on a Kaggle Kernel.**
# We need now to extract the train and test files from the zip. This is the code to use :

#%%
#We unzip the train and test zip file
archive_train = ZipFile("Data/train.zip", 'r')
archive_test = ZipFile("Data/test.zip", 'r')

#This line shows the 5 first image name of the train database
archive_train.namelist()[0:5]

#This line shows the number of images in the train database
len(archive_train.namelist()[:])-1 #we must remove the 1st value


#%%
10222

#%% [markdown]
# ### **3. Resize and normalize the data**
# 
# The next cells contains the following features:
# 
# - a function which ****create a pickle file to save all the images unzipped.
# - all the images do not have the same shape. For our model, we need to resize them to the same shape. We use the commonly practice to reshape them as a square.
# - we normalize our dataset by dividing by 255 all the pixel values. The new pixels values will be in the range [0,1].

#%%
# This function help to create  a pickle file gathering all the image from a zip folder
###############
def DataBase_creator(archivezip, nwigth, nheight, save_name):
    #We choose the archive (zip file) + the new wigth and height for all the image which will be reshaped
    
    # Start-time used for printing time-usage below.
    start_time = time.time()
    
    s = (len(archivezip.namelist()[:])-1, nwigth, nheight,3) #nwigth x nheight = number of features because images are nwigth x nheight pixels
    allImage = np.zeros(s)

    for i in range(1,len(archivezip.namelist()[:])):
        filename = BytesIO(archivezip.read(archivezip.namelist()[i]))
        image = PIL.Image.open(filename) # open colour image
        image = image.resize((nwigth, nheight))
        image = np.array(image)
        image = np.clip(image/255.0, 0.0, 1.0) #255 = max of the value of a pixel

        allImage[i-1]=image
    
    #we save the newly created data base
    pickle.dump(allImage, open( save_name + '.p', "wb" ) )
    
    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

#%% [markdown]
# **We define the new image size applied for all images :**

#%%
image_resize = 60


#%%
DataBase_creator(archivezip = archive_train, nwigth = image_resize, nheight = image_resize , save_name = "train")

#%% [markdown]
# Time usage: 0:00:40

#%%
DataBase_creator(archivezip = archive_test, nwigth = image_resize, nheight = image_resize , save_name = "test")

#%% [markdown]
# Time usage: 0:00:41
# 
# You have now a train and test pickle files. Next time you open this Python Notebook, you can load them directly and the step above can be skip if we relaunch the code later.

#%%
#load TRAIN
train = pickle.load( open( "train.p", "rb" ) )
train.shape


#%%
(10222, 60, 60, 3)


#%%
#load TEST
test = pickle.load( open( "test.p", "rb" ) )
test.shape


#%%
(10357, 60, 60, 3)

#%% [markdown]
# ## **The following code is design for the *Kaggle Kernel*. DO NOT COPY IT ON YOUR COMPUTER.**

#%%
df_train= pd.read_csv('../input/labels.csv')
df_train.sample(5)


#%%
image_resize = 60
nwigth = image_resize
nheight = image_resize


# Start-time used for printing time-usage below.
start_time = time.time()

s = (len(df_train['breed']), nwigth, nheight,3) #nwigth x nheight = number of features because images are nwigth x nheight pixels
allImage = np.zeros(s)
i= 0
for f, breed in df_train.values:
    image = PIL.Image.open('../input/train/{}.jpg'.format(f))
    image = image.resize((nwigth, nheight))
    image = np.array(image)
    image = np.clip(image/255.0, 0.0, 1.0) #255 = max of the value of a pixel
    i += 1
    
    allImage[i-1]=image

train = allImage

# Ending time.
end_time = time.time()

# Difference between start and end-times.
time_dif = end_time - start_time
print(time_dif)

#%% [markdown]
# ## **We come back to the code commonly used by the Kaggle Kernel or your jupyter notebook**

#%%
#let's check one image from the train data base
lum_img = train[100,:,:,:]
plt.imshow(lum_img)
plt.show()

#%% [markdown]
# ### **4. Zoom on label from the train dataset**
# 
# ## **The following code is design to be run on *your own JUPYTER NOTEBOOK made on your computer*. It will not work on a Kaggle Kernel.******

#%%
#######Upload the zip (input data base)########
labels_raw = pd.read_csv("Data/labels.csv.zip", compression='zip', header=0, sep=',', quotechar='"')

#Check 5 random values
labels_raw.sample(5)

#%% [markdown]
# ## **The following code is design for the *Kaggle Kernel*. DO NOT COPY IT ON YOUR COMPUTER.**

#%%
labels_raw = pd.read_csv('../input/labels.csv')
df_train.sample(5)

#%% [markdown]
# ## **We come back to the code commonly used by the Kaggle Kernel or your jupyter notebook**
#%% [markdown]
# ### **5. Extract the N most represented breeds**
# 
# We will reduce the data base so that we can reduce the complexity of our model.
# In addition, it will help for the calculation as there will be only N breeds to classify.
# We will be able to easily run the model in less than 10 minutes.

#%%
Nber_of_breeds = 8

#######Get the N most represented breeds########
def main_breeds(labels_raw, Nber_breeds , all_breeds='TRUE'):
    labels_freq_pd = itemfreq(labels_raw["breed"])
    labels_freq_pd = labels_freq_pd[labels_freq_pd[:, 1].argsort()[::-1]] #[::-1] ==> to sort in descending order
    
    if all_breeds == 'FALSE':
        main_labels = labels_freq_pd[:,0][0:Nber_breeds]
    else: 
        main_labels = labels_freq_pd[:,0][:]
        
    labels_raw_np = labels_raw["breed"].as_matrix() #transform in numpy
    labels_raw_np = labels_raw_np.reshape(labels_raw_np.shape[0],1)

    labels_filtered_index = np.where(labels_raw_np == main_labels)
    
    return labels_filtered_index

labels_filtered_index = main_breeds(labels_raw = labels_raw, Nber_breeds = Nber_of_breeds, all_breeds='FALSE')
labels_filtered = labels_raw.iloc[labels_filtered_index[0],:]
train_filtered = train[labels_filtered_index[0],:,:,:]

print('- Number of images remaining after selecting the {0} main breeds : {1}'.format(Nber_of_breeds, labels_filtered_index[0].shape))
print('- The shape of train_filtered dataset is : {0}'.format(train_filtered.shape))

#%% [markdown]
# Let's have a look on 1 image :

#%%
#print(labels_filtered[90])
lum_img = train_filtered[1,:,:,:]
plt.imshow(lum_img)
plt.show()

#%% [markdown]
# ### **6. One-hot labels**

#%%
#We select the labels from the N main breeds
labels = labels_filtered["breed"].as_matrix()
labels = labels.reshape(labels.shape[0],1) #labels.shape[0] looks faster than using len(labels)
labels.shape


#%%
#Function to breate one-hot labels
def matrix_Bin(labels):
    labels_bin=np.array([])

    labels_name, labels0 = np.unique(labels, return_inverse=True)
    labels0
    
    for _, i in enumerate(itemfreq(labels0)[:,0].astype(int)):
        labels_bin0 = np.where(labels0 == itemfreq(labels0)[:,0][i], 1., 0.)
        labels_bin0 = labels_bin0.reshape(1,labels_bin0.shape[0])

        if (labels_bin.shape[0] == 0):
            labels_bin = labels_bin0
        else:
            labels_bin = np.concatenate((labels_bin,labels_bin0 ),axis=0)

    print("Nber SubVariables {0}".format(itemfreq(labels0)[:,0].shape[0]))
    labels_bin = labels_bin.transpose()
    print("Shape : {0}".format(labels_bin.shape))
    
    return labels_name, labels_bin


#%%
labels_name, labels_bin = matrix_Bin(labels = labels)
labels_bin[0:9]

#%% [markdown]
# ### **7. Quick checks on Labels**
# 
# Let's see exactly the N labels we keep.
# As you will see below from the one-hot labels arry you can find which breed it corresponds.

#%%
for breed in range(len(labels_name)):
    print('Breed {0} : {1}'.format(breed,labels_name[breed]))


#%%
labels[0:9]


#%%
#You can proceed backward with np.argmax to find the breed of an image
labels_cls = np.argmax(labels_bin, axis=1)
labels_name[labels_cls[2]]


#%%
i=11
print(labels_name[labels_cls[i]])
lum_img = train_filtered[i,:,:,:]
plt.imshow(lum_img)
plt.show()

#%% [markdown]
# ## **Convolutional Neural Network**
# 
# ### **1. Creation of a Train and Validation DataBase**
# 
# We split our train data base in two: a train data base and a validation database.
# Therefore, we can check the accuracy of the model train made from the 'train database', on the validation database.
# 
# 

#%%
num_validation = 0.30
X_train, X_validation, y_train, y_validation = train_test_split(train_filtered, labels_bin, test_size=num_validation, random_state=6)
X_train.shape

#%% [markdown]
# *Here you will find a code which also split a database in two.*
# 
# #Creation of the Train DataBase and Test DataBase
# #x% of the observations will belong to the Train DataBase
# 
# def train_test_creation(x, data, toPred):
#     indices = sample(range(data.shape[0]),int(x * data.shape[0]))
#     indices = np.sort(indices, axis=None) 
#     index = np.arange(data.shape[0])
#     reverse_index = np.delete(index, indices,0)
#     
#     train_toUse = data[indices]
#     train_toPred = toPred[indices]
#     test_toUse = data[reverse_index]
#     test_toPred = toPred[reverse_index]
#         
#     return train_toUse, train_toPred, test_toUse, test_toPred
# 
# df_train_toUse, df_train_toPred, df_test_toUse, df_test_toPred = train_test_creation(0.7, train_filtered, labels_bin)
# df_train_toUse.shape

#%%
#from the one-hot database we can find the breed.
df_validation_toPred_cls = np.argmax(y_validation, axis=1)
df_validation_toPred_cls[0:9]

#%% [markdown]
# Quick check of the array created :

#%%
i=2
print(labels_name[df_validation_toPred_cls[i]])
print(df_validation_toPred_cls[i])
lum_img = X_validation[i,:,:,:]
plt.imshow(lum_img)
plt.show()

#%% [markdown]
# ### **2. CNN with Tensorflow - definition of the layers**
# 
# The CNN archtecture will be as followed :
# 
# - Convolutional Layer n°1 with 32 filters
#  + Max pooling
#  + Relu
# - Convolutional Layer n°2 with 64 filters
#  + Max pooling
#  + Relu
# - Convolutional Layer n°3 with 128 filters
#  + Max pooling
#  + Relu
#  + DropOut
# - Flatten Layer
# - Fully Connected Layer with 500 nodes
#  + Relu
#  + DropOut
# - Fully Connected Layer with n nodes (n = number of breeds)

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
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
#outputs random value from a truncated normal distribution

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))
#outputs the constant value 0.05

#%% [markdown]
# **I mainly reuse the code of Hvass and his comments. **If you need more information, just visit his github (I put the link on the top of this kernel). His works and sharing are really great, I strongly advise you to have a look.
# The main difference is that I added dropout.

#%%
def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True,
                   use_dropout=True):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)
    
    if use_dropout:
        layer = tf.nn.dropout(layer,keep_prob_conv)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights


#%%
def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()
    
    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features


#%%
def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True,
                 use_dropout=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)
    
    if use_dropout:
        layer = tf.nn.dropout(layer,keep_prob_fc)
        
    return layer

#%% [markdown]
# ### **3. CNN with Tensorflow - set up of placeholder tensor**

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
layer_conv1, weights_conv1 =     new_conv_layer(input=x_image,
                   num_input_channels=num_channels,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True,
                   use_dropout=False)
    
layer_conv2, weights_conv2 =     new_conv_layer(input=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=True,
                   use_dropout=False)
    
layer_conv3, weights_conv3 =     new_conv_layer(input=layer_conv2,
                   num_input_channels=num_filters2,
                   filter_size=filter_size3,
                   num_filters=num_filters3,
                   use_pooling=True,
                   use_dropout=True)


#%%
layer_flat, num_features = flatten_layer(layer_conv3)


#%%
#Train
layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True,
                         use_dropout=True)

layer_fc1

layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         use_relu=False,
                         use_dropout=False)

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

#function next_batch
def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


#%%
def optimize(num_iterations, X):
    global total_iterations
    
    start_time = time.time()
    
    #array to plot
    losses = {'train':[], 'validation':[]}
    
    for i in range(num_iterations):
            total_iterations += 1
            # Get a batch of training examples.
            # x_batch now holds a batch of images and
            # y_true_batch are the true labels for those images.
            x_batch, y_true_batch = next_batch(batch_size, X_train, y_train)

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


#%%
init_variables()
total_iterations = 0


#%%
optimize(num_iterations=3500, X=250)

#%% [markdown]
# As you can see, the model tends to overfit and is not very good. 
#%% [markdown]
# ### **7. CNN with Tensorflow - Results**
# 
# The results are not so good as the accuracy is only 44%. Using a pre-trained model with Keras will give you a better result but with this model, you will know how to build from scratch your own CNN with tensorflow.
# 
# By having more photos of dogs, we can increase the accuracy. in addition, we can create new images in our training dataset by rotating the images. it's what we call image augmentation. It will help the model to detect a patern which can have different 'position' in the space.
# 
# Go a 'bit' deeper on the result :
# 
# I just share some function to show some images from the new test database with the corresponding breeds and the predicted breeds.
# I also add the confusion matrix to see the results.

#%%
def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 12
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(4, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


#%%
def plot_confusion_matrix(data_pred_cls,data_predicted_cls):
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.
  
    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=data_pred_cls,
                          y_pred=data_predicted_cls)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

#%% [markdown]
# Launch the following codes, you will have 'nice' results!

#%%
feed_dict_validation = {x: X_validation,
                    y_true: y_validation,
                    keep_prob_conv : 1,
                    keep_prob_fc : 1}
df_validation_Predicted_cls = session.run(y_pred_cls, feed_dict=feed_dict_validation)

plot_images(images=X_validation[50:62],
            cls_true=df_validation_toPred_cls[50:62],
            cls_pred=df_validation_Predicted_cls [50:62])


#%%
i = 63
print(("True : {0} / {1}").format(df_validation_toPred_cls[i], labels_name[df_validation_toPred_cls[i]]))
print(("Pred : {0} / {1}").format(df_validation_Predicted_cls[i], labels_name[df_validation_Predicted_cls[i]]))

lum = X_validation[i,:,:,:]
plt.imshow(lum)
plt.show()


#%%
plot_confusion_matrix(df_validation_toPred_cls,df_validation_Predicted_cls)

#%% [markdown]
# As you can see, the model has difficulties to differenciate Breed 1 : bernese_mountain_dog and Breed 2 : entlebucher.
# This 2 breeds are look alike a lot (same color and shape). So, it's look normal that our model have make some mistakes between this two breeds.
#%% [markdown]
# ### **What the weights look like**
# 
# *Here we use a Hvass code to plot the weights and the layers. It's great his code and again go to visit his github (see the link on the top of this kernel)!

#%%
"Print the weights"
def plot_conv_layer(layer, image):
    feed_dict = {x: [image],
                keep_prob_conv : 1,
                keep_prob_fc : 1}

    values = session.run(layer, feed_dict=feed_dict)

    num_filters = values.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))
    
    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot the output images of all the filters.
    for i, ax in enumerate(axes.flat):
        # Only plot the images for valid filters.
        if i<num_filters:
            img = values[0, :, :, i]

            # Plot image.
            ax.imshow(img, interpolation='nearest', cmap='binary')
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.show()


#%%
def plot_conv_weights(weights, input_channel):

    w = session.run(weights)

    w_min = np.min(w)
    w_max = np.max(w)

    # Number of filters used in the conv. layer.
    num_filters = w.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))
    
    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i<num_filters:
            img = w[:, :, input_channel, i]

            # Plot image.
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.show()


#%%
image1 = train[0,:,:,:]
plt.imshow(image1)
plt.show()

#%% [markdown]
# ## **Layer 1**

#%%
plot_conv_weights(weights=weights_conv1, input_channel = 0)


#%%
plot_conv_weights(weights=weights_conv1, input_channel = 1)


#%%
plot_conv_weights(weights=weights_conv1, input_channel = 2)


#%%
plot_conv_layer(layer=layer_conv1, image=image1)

#%% [markdown]
# ## **Layer 2**

#%%
plot_conv_weights(weights=weights_conv2, input_channel = 0)


#%%
plot_conv_layer(layer=layer_conv2, image=image1)

#%% [markdown]
# ## **Layer 3**

#%%
plot_conv_weights(weights=weights_conv3, input_channel = 1)


#%%
plot_conv_layer(layer=layer_conv3, image=image1)

