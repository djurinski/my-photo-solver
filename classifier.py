from tensorflow.python.keras.models import model_from_json
import numpy as np

from prepareTraining import X_train, y_train, X_test, y_test

json_file = open('model_final3.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model_final3.h5")

# loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# score = loaded_model.evaluate(X_test, y_test, verbose=0)
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))



def classify(img):

    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=3)
    img = img.astype('float32')
    img /= 255

    prediction = loaded_model(img)
    result = np.argsort(prediction)
    result = result[0][::-1]
    from prepareTraining import label_encoder
    predicted_label = label_encoder.inverse_transform(np.array(result))

    return predicted_label[0]