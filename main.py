# Importing the relevant libraries
import tensorflow as tf
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

# Something something trying to get rid of annoying error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all logs, 1 = info, 2 = warning, 3 = error only

# Reading the data
train_labels = pd.read_csv("train_labels.csv")
# Show the first 5 rows
train_labels.head()
train_labels['target'].value_counts().plot.bar()
plt.show()

# Unzipping training data folder
with zipfile.ZipFile("images.zip", 'r') as zip_ref:
    zip_ref.extractall("images")

# List and select random image from training data
data = os.listdir("images")
sample = random.choice(data)

# Read image and convert to CVT colour
img = cv2.imread("images/"+sample)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(sample)
plt.imshow(img, cmap='gray')
plt.show()

train_labels["target"] = train_labels["target"].replace({0: 'unmask', 1: 'mask'})


print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("GPU devices:", tf.config.list_physical_devices('GPU'))

# Defining how data is passed to the input layer
image_size = 224
input_shape = (image_size, image_size, 3)
batch_size = 16

# Using VGG16 pre-trained convolution network
pre_trained_model = VGG16(input_shape=input_shape, include_top=False, weights="imagenet")
for layer in pre_trained_model.layers[:15]:
    layer.trainable = False
for layer in pre_trained_model.layers[15:]:
    layer.trainable = True
    last_layer = pre_trained_model.get_layer('block5_pool')
    last_output = last_layer.output
x = GlobalMaxPooling2D()(last_output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(2, activation='softmax')(x)
model = Model(pre_trained_model.input, x)
model.compile(loss='binary_crossentropy',
    optimizer=optimizers.SGD(learning_rate=1e-4, momentum=0.9),
    metrics=['accuracy'])
model.summary()

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
earlystop = EarlyStopping(patience=10)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',patience=2,verbose=1,factor=0.5,min_lr=0.00001)
callbacks = [earlystop, learning_rate_reduction]

from sklearn.model_selection import train_test_split
train_df,validate_df=train_test_split(train_labels,test_size=0.2,random_state=42)
train_df = train_df.reset_index(drop='True')
validate_df = validate_df.reset_index(drop='True')

train_df.head()
train_df['target'].value_counts().plot.bar()
plt.show()

validate_df['target'].value_counts().plot.bar()
plt.show()

# Generate batches of tensor image data with real-time data augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
# Categorical encodes categorical variables
from tensorflow.keras.utils import to_categorical

# Here we are formatting the training data
train_datagen = ImageDataGenerator(rotation_range=15,
                                 rescale=1./255,
                                 shear_range=0.1,
                                 zoom_range=0.2, # zoom range (1-0.2 to 1+0.2)
                                 horizontal_flip=True,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1)
train_generator = train_datagen.flow_from_dataframe(dataframe=train_df,
                                                  directory="images/",
                                                  x_col="image",
                                                  y_col="target",
                                                  target_size=(image_size,image_size),
                                                  class_mode='categorical',
                                                  batch_size=15)
# Here we are formatting images on the validation data
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(validate_df,
                                                  directory="images/",
                                                  x_col="image",
                                                  y_col="target",
                                                  target_size=(image_size,image_size),
                                                  class_mode='categorical',
                                                  batch_size=15)

epochs = 100
total_validate = validate_df.shape[0]
total_train = train_df.shape[0]
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate // batch_size,
    steps_per_epoch=total_train // batch_size,
    callbacks=callbacks,
)

# Here we are creating a list of pictures - we are appending images on the list.
# Our data source is the original data before splitting to test and train data
target = []
for i in data:
    flag = 0
    for j in train_df["image"]:    # ????
        if (i == j):
            flag = 1
            break
        else:
            continue
    if (flag == 0):
        target.append(i)

#creating a test dataframe with images and the target is umask for all images
test = pd.DataFrame({
    'image': target,
    'target':"unmask"
})
test.head()

test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    test,
    directory="images/",
    x_col="image",
    y_col="target",
    target_size=(image_size,image_size),
    class_mode='categorical',
    batch_size=15,
    shuffle=False)
nb_samples = test.shape[0]
predict = model.predict(test_generator, steps=np.ceil(nb_samples/batch_size))

# Here we are converting the submission data to a dataframe
test["target"]=predict
#here we are converting to a csv file
test.to_csv("submission.csv",index=False)


# Rezip and delete unused folder (Unused due to git and Github LFS characteristics)
#
# with zipfile.ZipFile("images.zip", 'w', zipfile.ZIP_DEFLATED) as zip_ref:
#     for root, _, files in os.walk("images"):
#         for file in files:
#             file_path = os.path.join(root, file)
#             zip_ref.write(file_path, os.path.relpath(file_path, "images"))
# shutil.rmtree("images")