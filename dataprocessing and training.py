import tensorflow as tf
from keras.models import  Model
from keras.layers import Input, Conv2D, \
    MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, Dropout, Lambda, BatchNormalization
from  keras.optimizers import Adam
from keras.layers import  Activation, MaxPool2D, concatenate
from keras.utils import  normalize
from sklearn.model_selection import  train_test_split
import cv2
import random
import  numpy as np
import glob
import keras.backend  as k
k.clear_session()
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Create 2 virtual GPUs with 1GB memory each
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=1024*3)
         ])
    logical_gpus = tf.config.list_logical_devices('GPU')

    print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)
rand = random.Random()
images = []
maskes = []
image_dir = "train imag/"
mask_dir = "train mask/"
size = 160
print("loading image")
image_name = glob.glob("train imag/*.jpg")
image_name.sort()
mask_name = glob.glob("train mask/*.png")
mask_name.sort()
for img in image_name:
    image = cv2.imread(img,0)
    image = cv2.resize(image, (160, 160))
    images.append(image)
for mask in mask_name:
    imag = cv2.imread(mask, 0)
    imag = cv2.resize(imag, (160, 160))
    thresh = 35
    imag = cv2.threshold(imag, thresh, 255, cv2.THRESH_BINARY)[1]
    maskes.append(imag)
imagedata = np.array(images)
imagedata = np.expand_dims(imagedata, 3)
maskdata = np.array(maskes)
maskdata = np.expand_dims(maskdata, 3)
maskdata = normalize(maskdata)
imagedata = imagedata / 255
print(imagedata.shape)
print(maskdata.shape)
print('max', imagedata.max())
print('label', np.unique(maskdata))
x_train, x_test, y_train, y_test = train_test_split(imagedata, maskdata, test_size=0.2, random_state=52)


def conv(input, filter):
    x = Conv2D(filter, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filter, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x
def encod(input, filter):
    x = conv(input, filter)
    p = MaxPool2D((2,2))(x)
    return x,p
def decod(input, skip, filter):
    x = Conv2DTranspose(filter, (2,2), strides=2, padding="same")(input)
    x = concatenate([x, skip])
    x = conv(x, filter)
    return x
def net(input_shape, nclass):
    inputs = Input(input_shape)
    s1, p1 = encod(inputs, 64)
    s2, p2 = encod(p1, 128)
    s3, p3 = encod(p2, 256)
    s4, p4 = encod(p3, 512)
    b1 = conv(p4, 1024)
    d1 = decod(b1, s4, 512)
    d2 = decod(d1, s3, 256)
    d3 = decod(d2, s2, 128)
    d4 = decod(d3, s1, 64)
    if nclass == 1:
        activation = 'sigmoid'
    else:
        activation = 'softmax'
    outputs = Conv2D(nclass, 1, padding="same",activation=activation)(d4)
    print(activation)
    model = Model(inputs, outputs)
    return model

imh = imagedata.shape[1]
imw = imagedata.shape[2]
imc = imagedata.shape[3]
input_shape = (imh, imw, imc)
print(imagedata.shape[1])
print(imagedata.shape[2])
print(imagedata.shape[3])

model = net(input_shape, 1)
model.compile(optimizer=Adam(learning_rate=0.05), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(x_train[0:2000],y_train[0:2000],batch_size=5,verbose=1,epochs=20,validation_data=(x_test,y_test),shuffle=False)

np.save('history.npy', history.history)
model.save("save.h5")