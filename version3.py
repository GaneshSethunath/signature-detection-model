import random
import gc
import os
import cv2
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns

import tensorflow as tf
from keras import layers
from keras import models
from keras.applications import InceptionResNetV2
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split

##importing the pre trained model
conv_base = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(150,150,3))


#defining the useful stuff i.e model architecture

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
##freezing pre trained layers
conv_base.trainable = False

##image parameters
nrows=150
ncolumns=150
channels=3

##image and label arrays
X=[]
y=[]


#loading the data

train_sig_dir=r"C:\Users\Ganesh Sethunath\Downloads\sigcomp-2009\train_genuine"
train_img_dir=r"C:\Users\Ganesh Sethunath\Downloads\sigcomp-2009\forgeries"

train_sig=[r"C:\Users\Ganesh Sethunath\Downloads\sigcomp-2009\train_genuine\{}".format(i) for i in os.listdir(train_sig_dir)]
train_imgs=[r"C:\Users\Ganesh Sethunath\Downloads\sigcomp-2009\forgeries\{}".format(i) for i in os.listdir(train_img_dir)]

random.shuffle(train_sig)
random.shuffle(train_imgs)

train=train_sig[:]+train_imgs[:400]

test=[r"C:\Users\Ganesh Sethunath\Downloads\sigcomp-2009\train_genuine\NFI-00201002.png",r"C:\Users\Ganesh Sethunath\Downloads\sigcomp-2009\train_genuine\NFI-00206002.png",r"C:\Users\Ganesh Sethunath\Downloads\sigcomp-2009\forgeries\NFI-00303002.png",r"C:\Users\Ganesh Sethunath\Downloads\sigcomp-2009\forgeries\NFI-00305002.png",
r"C:\Users\Ganesh Sethunath\Downloads\sigcomp-2009\forgeries\NFI-00301002.png",r"C:\Users\Ganesh Sethunath\Downloads\sigcomp-2009\forgeries\NFI-00302002.png",
r"C:\Users\Ganesh Sethunath\Downloads\sigcomp-2009\forgeries\NFI-00304002.png",r"C:\Users\Ganesh Sethunath\Downloads\sigcomp-2009\forgeries\NFI-00306002.png"]

random.shuffle(train)


##garbage collection
del train_sig
del train_imgs

gc.collect()

##image processing and labelling function
def process_img(images):
	imgs=[]
	labels=[]
	for image in images:
		if 'NFI-002' in image:
			labels.append(1)
		else:
			labels.append(0)
		imgs.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (nrows,ncolumns), interpolation=cv2.INTER_CUBIC))			

	return imgs,labels

X,y=process_img(train)	


X=np.array(X)
y=np.array(y)

print(X.shape,y.shape)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=2)

##some important parameters
ntrain = len(X_train)
nval = len(X_val)
batch_size = 32  

print("\n\n\n",nval,"\n\n\n")
##garbage collection
del X
del y
gc.collect()


#image preparation

train_datagen = ImageDataGenerator(rescale=1./255,   #Scale the image between 0 and 1
                                    rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale=1./255)

##Create the image generators
train_generator = train_datagen.flow(X_train, y_train,batch_size=batch_size)
val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)

#model compiling

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=2e-5), metrics=['acc'])
model.summary()

#model training

history = model.fit_generator(train_generator,
                              steps_per_epoch=ntrain // batch_size,
                              epochs=10,
                              validation_data=val_generator,
                              validation_steps=nval // batch_size)

#saving the model

model.save_weights('model_wieghts.h5')
model.save('model_keras.h5')


#testing trained model

X_test, y_test = process_img(test) #Y_test in this case will be empty.
x = np.array(X_test)
test_datagen = ImageDataGenerator(rescale=1./255)

##testing parameters
total=0
index=0
text_labels=[]

##prediction
for batch in train_datagen.flow(x, batch_size=1):
    pred = model.predict(batch)
    total+=1
    text_labels.append(pred[0])
    if total==6:
      break

    
    
for i in range(6):

	print(y_test[i]," --> ",text_labels[i][i])
