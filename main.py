# Importing the relevant libraries
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import random
import zipfile
import os
import shutil
from IPython.display import Image as ShowImage
# Keras libraries
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization,GlobalMaxPooling2D
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from keras.models import Model

# Reading the data
train_labels = pd.read_csv("train_labels.csv")
# Show the first 5 rows
train_labels.head()

train_labels['target'].value_counts().plot.bar()
plt.show()

with zipfile.ZipFile("images.zip", 'r') as zip_ref:
    zip_ref.extractall("images")

data = os.listdir("images")

sample = random.choice(data)

img = cv2.imread("images/"+sample)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(sample)
plt.imshow(img, cmap='gray')
plt.show()


# with zipfile.ZipFile("images.zip", 'w', zipfile.ZIP_DEFLATED) as zip_ref:
#     for root, _, files in os.walk("images"):
#         for file in files:
#             file_path = os.path.join(root, file)
#             zip_ref.write(file_path, os.path.relpath(file_path, "images"))
# shutil.rmtree("images")