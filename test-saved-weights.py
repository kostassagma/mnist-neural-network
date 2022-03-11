import tensorflow as tf
from tensorflow import keras
import numpy as np
from utils import load_custom, load_mnist, view_img

custom_imgs = load_custom()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.load_weights('saved_model/')

prediction = model.predict(custom_imgs)

for i, e in enumerate(prediction):
    view_img(custom_imgs[i], xlabel=f"Custom Images", label=f"Prediction: {np.argmax(prediction[i])}")


