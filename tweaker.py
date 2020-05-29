import numpy as np
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

# Load Model

(train_X , train_y), (test_X , test_y) = mnist.load_data("mymnist.data")
# Reshape data and change type
test_X = test_X.reshape(-1 , 28, 28, 1)
train_X = train_X.reshape(-1 ,  28, 28, 1)
test_X = test_X.astype("float32")
train_X = train_X.astype("float32")
# One hot encoding
test_y = npu.to_categorical(test_y)
train_y = npu.to_categorical(train_y)

accuracy= open("accuracy.txt","r")
accuracy = float(accuracy.read())
accuracy = accuracy *100
#Initials
neurons = 10
epochs = 1
test = 1
flag = 0
kernel = 8
batch_size = 128
#filter = 3


while int(accuracy)<85:
    if flag == 1 :
        model = keras.backend.clear_session()
        neurons = neurons+10
        epochs = epochs+1
        test = test + 1
        kernel = kernel * 2
        test = test + 1
    print("* * * TRIAL : ",test ,"-----------------")
    model=Sequential()
    model.add(Conv2D(kernel, (3,3), input_shape = (28, 28, 1), activation = 'relu'))
    model.add(MaxPooling2D(pool_size =(2,2)))
    model.add(Flatten())
    model.add(Dense(neurons, activation = 'relu'))
    model.add(Dense(10, activation = 'softmax'))
    model.compile( optimizer= "Adam" , loss='categorical_crossentropy',  metrics=['accuracy'] )
    train_X.shape
    model_predict= model.fit(train_X, train_y,batch_size=batch_size,verbose=1,epochs=epochs,validation_data=(test_X, test_y),shuffle=True)
    output = model.evaluate(test_X, test_y, verbose=False)
    print('Test loss :', output[0]*100)
    print('-------Accuracy of the model :', output[1]*100)
    accuracy = output[1]*100
    print("_______________________________________________________")
    print()
    print()
    flag = 1

print("Total numbers of epochs :" , epochs)
print("Total number of filters :", kernel)
print("Total number of neurons :", neurons)
print("Final Accuracy : ", accuracy)


import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
host_address="arushishukla09@gmail.com"
host_pass="swift9415581194papa"
guest_address="arushishukla09@gmail.com"
subject="Model Accuracy"
content='''Hi,developer
            You have achieved your desired accuracy for this model
            Accuracy page is attached with this mail'''
message=MIMEMultipart()
message['From']=host_address
message['To']=guest_address
message['Subject']=subject
message.attach(MIMEText(content,'plain'))





#attaching files
from email.mime.base import MIMEBase
from email import encoders
filename="prg.py"
attachment=open('/home/prg.py', 'rb')
p=MIMEBase('application','octet-stream')
p.set_payload((attachment).read())
encoders.encode_base64(p)
p.add_header('Content-Disposition',"attachment; filename= %s" % filename)
message.attach(p)
#attach finished





message.attach(MIMEText('accuracy.txt', 'plain'))
session=smtplib.SMTP("smtp.gmail.com",587)
session.starttls()
session.login(host_address,host_pass)
text=message.as_string()
session.sendmail(host_address,guest_address,text)
session.quit()
print('Successfuly Sent')
