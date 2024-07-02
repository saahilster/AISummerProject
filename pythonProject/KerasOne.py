import os
os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
print(tf.__version__)

tf.keras.datasets.mnist.load_data(
    path='mnist.npz'
)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train, x_test = x_train / 255, x_test / 255

model = keras.Sequential(
    [
        # breaks down 28x28 grid to 1d array for densing
        keras.layers.Flatten(input_shape=(28, 28)),
        # Dense creates the amount of neurons connected to the first layer and activation for neurons
        keras.layers.Dense(128, activation='relu'),
        # prevents overfitting
        keras.layers.Dropout(0.2),
        # final layer making sure no value above 10 units with softmax
        keras.layers.Dense(10, activation='softmax')
    ]
)



model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=2)
model.evaluate(x_test, y_test, batch_size=32, verbose=2)

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test)
model.summary()
