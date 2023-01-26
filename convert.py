import tensorflow as tf
from tensorflow import keras

model=keras.models.load_model('model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('model.tflite','wb') as f_out:
    f_out.write(tflite_model) 