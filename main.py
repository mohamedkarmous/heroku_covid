import os

import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

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


def get_proba(l):
    if min(l) < 0:
        for i in range(len(l)):
            l[i] = l[i] + abs(min(l))
    l2 = [round((i / sum(l)) * 100) for i in l]
    return (l2)

class _CovidClassifier:
    model = None
    _mapping = ["COVID", "NORMAL", "PNEUMONIA"]
    _instance = None



    def predict(self, image_path):
        img = image.load_img(image_path, target_size=(256, 256))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        layer_name = 'dense_19'
        intermediate_layer_model = Model(inputs=self.model.input,
                                         outputs=self.model.get_layer(layer_name).output)
        intermediate_output = intermediate_layer_model.predict(img)
        p = self.model.predict(img)
        print("class probabilities : ",get_proba(intermediate_output[0]))
        print("prediction : ",self._mapping[np.argmax(p)])
        proba= get_proba(intermediate_output[0])
        result2=[]
        for i in range(len(proba)):
            result2+=[(self._mapping[i],proba[i])]
        result=[self._mapping[np.argmax(p)]+" "++str(proba[np.argmax(p)])+"%",result2]
        return (result)


def CovidClassifier():
    if _CovidClassifier._instance is None:
        _CovidClassifier._instance = _CovidClassifier()
        _CovidClassifier.model = tf.keras.models.load_model('my_model_3.h5')
    return _CovidClassifier._instance
