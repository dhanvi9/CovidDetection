# -*- coding: utf-8 -*-
"""FinalCovidDetection.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1H6QL3MnrpSOkj21peA6UOVKTXacOnmj2
"""

from google.colab import drive
drive.mount('/content/gdrive')

TRAIN_PATH = "/content/gdrive/MyDrive/COVID_DATASET/Train"
VAL_PATH = "/content/gdrive/MyDrive/COVID_DATASET/Val"

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.models import * 
from keras.preprocessing import image

# CNN Based Model in Keras

model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(224,224,3))) #images are rgb
model.add(Conv2D(64,(3,3),activation='relu')) #increases non-lineraity ('relu') and reduce the number of paramters
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss=keras.losses.binary_crossentropy,optimizer='adam',metrics=['accuracy'])

model.summary()

# Train from scratch
from skimage import exposure
def HE(img):
  #print(img)
  img=img/255.0
  # plt.imshow(img)
  # plt.axis("off")
  # plt.show()
  #print(img.shape)
  img_eq = np.arange(150528,dtype=float).reshape(224,224,3)
  img_eq[:,:,0] = exposure.equalize_hist(img[:,:,0])
  img_eq[:,:,1] = exposure.equalize_hist(img[:,:,1])
  img_eq[:,:,2] = exposure.equalize_hist(img[:,:,2])
  #img_eq=np.array(img_eq)
  # plt.imshow(img_eq)
  # plt.axis("off")
  # plt.show()

  #print(img_eq.shape)
  return img_eq*255.0



train_datagen = image.ImageDataGenerator(
    rescale = 1.0/255,  #normalization leads to early convergence our wrights starts from 0 so reaching 0.0 to 0.XX is easy
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    rotation_range=90,
    preprocessing_function=HE
    
)

val_dataset = image.ImageDataGenerator(rescale=1.0/255,preprocessing_function=HE)
test_dataset = image.ImageDataGenerator(rescale=1.0/255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    rotation_range=90,
    preprocessing_function=HE)

train_generator = train_datagen.flow_from_directory(
    '/content/gdrive/MyDrive/COVID_DATASET/Train',
    target_size = (224,224),
    batch_size = 32,
    class_mode = 'binary'
    )

validation_generator = val_dataset.flow_from_directory(
    '/content/gdrive/MyDrive/COVID_DATASET/Val',
    target_size = (224,224),
    batch_size = 32,
    class_mode = 'binary'
    )


test_generator = test_dataset.flow_from_directory(
    '/content/gdrive/MyDrive/COVID_DATASET/Test',
    target_size = (224,224),
    batch_size = 32,
    shuffle=False,
    class_mode = 'binary')

plt.figure(figsize=(20,20))
for x,y in train_generator:
  print(x.shape,y.shape)
  for i in range(32):
    plt.subplot(6,6,i+1)
    plt.imshow(x[i])
    plt.axis("off")
  plt.show()
  break


plt.figure(figsize=(20,20))
for x,y in validation_generator:
  print(x.shape,y.shape)
  for i in range(32):
    plt.subplot(6,6,i+1)
    plt.imshow(x[i])
    plt.axis("off")
  plt.show()
  break


plt.figure(figsize=(20,20))
for x,y in test_generator:
  print(x.shape,y.shape)
  for i in range(32):
    plt.subplot(6,6,i+1)
    plt.imshow(x[i])
    plt.axis("off")
  plt.show()
  break

train_generator.class_indices

from keras.callbacks import ModelCheckpoint #saves the best model, also reduces overfitting
checkpoint = ModelCheckpoint(filepath="model1.h5",
    save_weights_only=False,
    verbose=0,
    monitor='val_accuracy',
    mode='auto',
    period=1,
    save_best_only=True)
hist = model.fit_generator(
    train_generator,
    steps_per_epoch=9,
    epochs = 50,
    validation_data = validation_generator,
    validation_steps=3,
    callbacks=[checkpoint]
)

acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss= hist.history['val_loss']

plt.style.use("seaborn")
plt.plot(acc,label="train acc")
plt.plot(val_acc,label="val acc")
plt.plot(loss,label='train loss')
plt.plot(val_loss,label='val loss')
plt.xlabel("No. of epochs",fontsize = 15)
plt.ylabel("Evaluation Metrics Value",fontsize = 15)
plt.legend()
plt.show()

from keras.models import load_model
model= load_model('/content/model1.h5')
model.evaluate(train_generator)
model.evaluate(validation_generator)

model.evaluate(test_generator)
model.predict(test_generator)



import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.models import * 
from keras.preprocessing import image
import os

img = image.load_img("/content/gdrive/MyDrive/image (1).png",target_size=(224,224))
img = image.img_to_array(img)
#img=img/255.0
img1 = np.expand_dims(img,axis=0)
p=model.predict_classes(img1)
print("______________",model.predict(img1))

img = HE(img)
img=img/255.0
img = np.expand_dims(img,axis=0)
p=model.predict_classes(img)
print("with HE",model.predict(img)[0,0])
print(p[0,0])







y_actual = []
y_test = []

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.models import * 
from keras.preprocessing import image
import os
for i in os.listdir("/content/gdrive/MyDrive/COVID_DATASET/Test/Normal/"):
  img = image.load_img("/content/gdrive/MyDrive/COVID_DATASET/Test/Normal/"+i,target_size=(224,224))
  img = image.img_to_array(img)
  img = HE(img)
  img=img/255.0
  # print("img_dhape",img.shape)
  img = np.expand_dims(img,axis=0)
  # print("ime",img.shape)
  #print(model.predict(img))
  p=model.predict_classes(img)
  y_test.append(p[0,0])
  y_actual.append(1)

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.models import * 
from keras.preprocessing import image
for i in os.listdir("/content/gdrive/MyDrive/COVID_DATASET/Test/Covid/"):
  img = image.load_img("/content/gdrive/MyDrive/COVID_DATASET/Test/Covid/"+i,target_size=(224,224))
  img = image.img_to_array(img)
  img = HE(img)
  img=img/255.0
  # print("img_dhape",img.shape)
  img = np.expand_dims(img,axis=0)
  p=model.predict_classes(img)
  y_test.append(p[0,0])
  y_actual.append(0)

y_actual = np.array(y_actual)
y_test = np.array(y_test)

from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_actual,y_test)

sns.heatmap(cm, cmap="plasma",annot=True)
plt.xlabel("Predicted labels",fontsize = 15)
plt.ylabel("True labels", fontsize = 15)
plt.title("Confusion matrix for Test Dataset",fontsize = 15)
plt.show()

from sklearn.metrics import classification_report

print(classification_report(y_actual, y_test))

model.fit_generator?





