import numpy as np
import cv2
import glob
from keras.models import load_model
from skimage.io import imread, imshow
import matplotlib.pyplot as plt

images = []

image_name = glob.glob("testimg/*.jpg")
for img in image_name:
    image = cv2.imread(img, 0)
    image = cv2.resize(image, (160, 160))
    images.append(image)
images = np.array(images)
images = np.expand_dims(images, 3)
images = images/255

model = load_model('save.h5')
result = model.predict(images)


for i in range(len(result)):
    imshow(images[i])
    plt.show()
    imshow(np.squeeze(result[i]))
    plt.show()




