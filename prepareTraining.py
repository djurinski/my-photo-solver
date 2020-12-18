import os
import cv2 as cv
import numpy as np
from sklearn import preprocessing

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from croppTool import croppImage

symbols_list = ['1', '2', '3', '4', '5', '6',
                '7', '8', '9', '0', '-', '+',
                "div", 'x', "lbr", "rbr"]

train_dataset = "train_set"

train_image = []
train_label = []

for symbols_dir in os.listdir(train_dataset):
    if symbols_dir.split()[0] in symbols_list:
        for image in os.listdir(train_dataset + "/" + symbols_dir):
            train_label.append(symbols_dir.split()[0])
            train_image.append(train_dataset + "/" + symbols_dir + "/" + image)

evaluate_set = "evaluation_set"
test_image = []
test_label = []

for symbols_dir in os.listdir(evaluate_set):
    if symbols_dir.split()[0] in symbols_list:
        for image in os.listdir(evaluate_set + "/" + symbols_dir):
            test_label.append(symbols_dir.split()[0])
            test_image.append(evaluate_set + "/" + symbols_dir + "/" + image)


X_train = []
X_test = []


# # loading the images from the path
# #UNCOMENT THIS SECTION IF TRAINIG NEW MODEL
# for path in train_image:
#     img = cv.imread(path)
#     img = croppImage(img)
#     img = cv.resize(img, (100, 100))
#     img = np.array(img)
#     img = np.expand_dims(img, axis=2)
#     X_train.append(img)
# #
# for path in test_image:
#     img = cv.imread(path)
#     img = croppImage(img)
#     img = cv.resize(img, (100, 100))
#     img = np.array(img)
#     img = np.expand_dims(img, axis=2)
#     X_test.append(img)

# creating numpy array from the images
X_train = np.array(X_train)
X_test = np.array(X_test)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

label_encoder = preprocessing.LabelEncoder()
y_train_temp = label_encoder.fit_transform(train_label)
y_test_temp = label_encoder.fit_transform(test_label)

y_train = keras.utils.to_categorical(y_train_temp, 16)
y_test = keras.utils.to_categorical(y_test_temp, 16)

model = Sequential()

# 1st layer and taking input in this of shape 100x100x3 ->  100 x 100 pixles and 3 channels
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(100, 100, 1), activation="relu"))
model.add(Conv2D(32, (3, 3), activation="relu"))

# maxpooling will take highest value from a filter of 2*2 shape
model.add(MaxPooling2D(pool_size=(2, 2)))

# it will prevent overfitting by making it hard for the model to idenify the images
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))

# last layer predicts 16 labels
model.add(Dense(16, activation="softmax"))

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer="adam",
    metrics=['accuracy']
)


