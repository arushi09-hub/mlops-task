#!/usr/bin/env python
# coding: utf-8

# In[23]:


import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.datasets import mnist
from keras.utils import np_utils as npu
from keras.backend import clear_session


# In[30]:


(x_train, y_train), (x_test, y_test)  = mnist.load_data()
x_train.shape

img = x_train[0].shape
print("size of image",img)
xTrain = xTrain.reshape(-1, 28, 28, 1)
xTest = xTest.reshape(-1, 28,28 , 1)
xTrain = xTrain.astype('float32')
xTest = xTest.astype('float32')
yTrain = npu.to_categorical(y_train)
yTest = npu.to_categorical(y_test)


# In[31]:


model = Sequential()

kernel = 8
filter = 3
pool = 2


# In[32]:


#our layers
model.add(Conv2D(kernel, (filter,filter), input_shape = (28, 28, 1), activation = 'relu'))

model.add(MaxPooling2D(pool_size =(pool,pool)))

model.add(Flatten())

model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))
model.compile( optimizer= "Adam" , loss='categorical_crossentropy',  metrics=['accuracy'] )

print(model.summary())


# In[33]:


batch_size = 128
epochs = 1

history = model.fit(xTrain, yTrain,
          batch_size=batch_size,verbose=1,
          epochs=epochs,
          validation_data=(xTest, yTest),
          shuffle=True)

model.save("mnist_LeNet.h5")


# In[35]:


output = model.evaluate(xTest, yTest, verbose=False)
print('Test loss:', output[0])
print('Test accuracy:', output[1])
accuracy = scores[1]

f= open("accuracy.txt","w+")
f.write(str(accuracy))
f.close()
print("Accuracy is : " , accuracy ,"%")


# In[ ]:




