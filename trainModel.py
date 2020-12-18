
import matplotlib.pyplot as plt
from prepareTraining import *
from prepareTraining import X_train, y_train

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


history = model.fit(
    X_train,
    y_train,
    batch_size=50,
    epochs=10,
    validation_split=0,
    shuffle=True
)


model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = model.evaluate(X_train, y_train, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))

#store model for future use
model_json = model.to_json()
with open("new_model1.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("new_model1.h5")
print("Model saved in memory")

fig, axs = plt.subplots(10, 10, figsize=[20, 10])
count = 0





for i in range(10):
    for j in range(10):
        image = cv.imread(test_image[int(len(test_image) / 100 * count +1)])
        img = croppImage(image)
        img = cv.resize(img, (100, 100))
        img = np.array(img)
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=3)
        img = img.astype('float32')
        img /= 255


        prediction = model(img)
        result = np.argsort(prediction)
        result = result[0][::-1]
        predicted_label = label_encoder.inverse_transform(np.array(result))

        axs[i][j].imshow(image)
        axs[i][j].set_title(str("Predicted: " + predicted_label[0]))

        count += 1
plt.show()

