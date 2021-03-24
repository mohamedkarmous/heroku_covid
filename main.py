import os

import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing import image
image_path="covid2.jpg"
"""
model = tf.keras.models.load_model('my_model2.hdf5')

# Check its architecture
model.summary()

img = image.load_img(image_path, target_size=(256, 256))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
p = model.predict(img)
classes=["COVID","NORMAL"]
print(classes[int(p[0][0])])
"""

class _CovidClassifier:
    model= None
    _mapping = [
        "COVID",
        "NORMAL"
    ]
    _instance = None

    def predict(self,image_path):
        img = image.load_img(image_path, target_size=(256, 256))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        p = self.model.predict(img)
        classes = ["COVID", "NORMAL"]
        return (self._mapping[int(p[0][0])])

def CovidClassifier():
    if _CovidClassifier._instance is None:
        _CovidClassifier._instance = _CovidClassifier()
        _CovidClassifier.model = tf.keras.models.load_model('my_model2.h5')
    return _CovidClassifier._instance
