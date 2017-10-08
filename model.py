import csv
import keras
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten , Cropping2D
from keras.layers import Lambda,Conv2D, MaxPooling2D, ZeroPadding2D, GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from sklearn.model_selection import StratifiedShuffleSplit
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
from cv2 import *
import os
from keras.callbacks import Callback,LearningRateScheduler
import math
import sklearn
import random
import sys
import multiprocessing
from scipy.misc import imread, imresize
import matplotlib.image as mpimg
from skimage import color

#### Step 1 read the metadata file ####
dir_loc = 'data/data/IMG/'   # image file location
shape = (66,200,3)           # new resize image
steering_correction = 0.15   # +/- angle correction
nbatch=128                   # size of batch for trainging your model
nepochs=10                    # number of epochs

samples = []

with open('data/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

samples.pop(0)
print (samples[0:5])
sklearn.utils.shuffle( samples )

# generate the train and validate samples
train_samples, validation_samples = train_test_split(samples, test_size=0.1,random_state=2341)



#### Step 2 Image preprocessing ####

def crop_resize(img):
    ''' function to crop the top and bottom parts and then resize the
      images to smaller_scale '''

    image = imresize(img[50:140,:,:],shape)
    return image

def flip(img):
    ''' function to flip the image '''

    image = np.copy(np.fliplr(img))
    return image


def rgb2yuv(image):
      
    m = np.array([[ 0.29900, -0.16874,  0.50000],
                 [0.58700, -0.33126, -0.41869],
                 [ 0.11400, 0.50000, -0.08131]])
      
    yuv = np.dot(image,m)
    yuv[:,:,1:]+=128.0
    return yuv


def preprocess(image):
    ''' performs image processing '''
    image = rgb2yuv(image)
    image = crop_resize(image)
    return image



#### Step 3  build a generator function to stream images ####


def generator(samples, batch_size=32):
    num_samples = len(samples)
    #print('generator started with size: ',len(samples)) 
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
     
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
         
            images = []
            angles = []
            for batch_sample in batch_samples:

                # choose a random image (center,left,right)
                idx = random.randrange(3)
                image = imread(dir_loc + batch_sample[idx].split('/')[-1])
                #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                angle = float(batch_sample[3])
                
                # image preprocessing/augmentation
                image = preprocess(image)
                
                if (idx==1):
                    angle = angle + steering_correction
                elif (idx==2):
                    angle = angle - steering_correction
                else:
                    angle = angle
                
                # randomm flip an image
                if (random.randrange(2))==1:
                    image = flip(image)
                    angle = angle*-1.0
                
                images.append(image)
                angles.append(angle)
        
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            #print(X_train.shape)
            yield sklearn.utils.shuffle(X_train, y_train)


# compile and train the model using the generator function


train_generator = generator(train_samples, batch_size=nbatch)
validation_generator = generator(validation_samples, batch_size=nbatch)



#### Step 3  train the model #####


model=Sequential()

model.add(Lambda(lambda x: x/255. - 0.5, input_shape=shape))
model.add( Conv2D( 24,kernel_size=(5, 5), strides=(2,2), activation = 'relu' ) )
model.add( Conv2D( 36,kernel_size=(5, 5), strides=(2,2), activation = 'relu' ) )
model.add( Conv2D( 48,kernel_size= (5, 5), strides=(2,2), activation = 'relu' ) )
model.add( Conv2D( 64, kernel_size=(3, 3), strides=(1,1), activation = 'relu' ) )
model.add( Conv2D( 64, kernel_size=(3, 3), strides=(1,1), activation = 'relu' ) )

model.add(Flatten())

model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))

model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
print(model.summary())

#model.fit_generator(train_generator,steps_per_epoch=len(train_samples)/nbatch,epochs=nepochs,validation_data=validation_generator,nb_val_samples=len(validation_samples)/nbatch,pickle_safe=False,nb_worker=1)

model.fit_generator(train_generator,steps_per_epoch=len(train_samples)/nbatch,epochs=nepochs,validation_data=validation_generator,nb_val_samples=len(validation_samples)/nbatch) 


###save the model####
model.save('model.h5')

json_string = model.to_json()
with open('model.json', 'w') as f:
    f.write(json_string)
