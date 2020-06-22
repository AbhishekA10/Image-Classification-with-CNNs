#Importing libraries
import tensorflow as tf
import keras
from keras import backend as K
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt

#Importing cifar10 dataset 
(x_train,y_train),(x_test,y_test)=tf.keras.datasets.cifar10.load_data()

#Printing the training and testing data shape
print("Training set:",x_train.shape,y_train.shape)
print("Testing set:",x_test.shape,y_test.shape)

#Defining the class labels
num=10
classes=['aeroplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

#Normalizing the training dataset
x_train2 = (x_train/255)-0.5
x_test2 = (x_test/255)-0.5
y_train2 = keras.utils.to_categorical(y_train,num)
y_test2 = keras.utils.to_categorical(y_test,num)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dropout,Flatten,Input,Dense,Activation

#Building the model
def make_model():
  model=Sequential()
  model.add(Input(shape=(32,32,3)))
  model.add(Conv2D(32,3,activation='relu',padding='same'))
  model.add(Conv2D(32,3,activation='relu'))
  model.add(MaxPooling2D(pool_size=2))
  model.add(Dropout(0.25))
  model.add(Conv2D(64,3,activation='relu',padding='same'))
  model.add(Conv2D(64,3,activation='relu'))
  model.add(MaxPooling2D(pool_size=2))
  model.add(Dropout(0.25))
  model.add(Conv2D(128,3,activation='relu',padding='same'))
  model.add(Conv2D(128,3,activation='relu'))
  model.add(MaxPooling2D(pool_size=2))
  model.add(Dropout(0.25))
  model.add(Flatten())
  model.add(Dense(num,activation='softmax'))
  model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
  return model

model=make_model()
model.summary()

#Training the model
model.fit(
    x_train2, y_train2,
    batch_size=32,
    epochs=10,
    validation_data=(x_test2, y_test2),
    callbacks =[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=3)])

#Predicting output
y_pred_test = model.predict(x_test2)
y_pred_test_classes = np.argmax(y_pred_test, axis=1)

cols = 5
rows = 3
fig = plt.figure(figsize=(2 * cols - 1, 3 * rows - 1))
for i in range(cols):
    for j in range(rows):
        random_index = np.random.randint(0, len(y_test))
        ax = fig.add_subplot(rows, cols, i * rows + j + 1)
        ax.grid('off')
        ax.axis('off')
        ax.imshow(x_test[random_index, :])
        pred_label =  classes[y_pred_test_classes[random_index]]
        true_label = classes[y_test[random_index, 0]]
        ax.set_title("pred: {}\ntrue: {}".format(
               pred_label, true_label
        ))
plt.show()

