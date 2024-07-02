import os

import numpy as np

os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
print(tf.__version__)

tf.keras.datasets.mnist.load_data(
    path='mnist.npz'
)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train, x_test = x_train / 255, x_test / 255

index = 1000
singleImageX = x_test[index]
singleImageY = y_test[index]

singleImageX = singleImageX.reshape(1, 28, 28)

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
# Compile the model with the correct optimizer parameter
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test)
model.summary()

predictions = model.predict(singleImageX)
predictions_label = np.argmax(predictions[0])

# Output the result
print(f"Predicted label: {predictions_label}")
print(f"True label: {singleImageY}")

# Display the image
plt.imshow(singleImageX[0], cmap='gray')
plt.title(f"Predicted: {predictions_label}, True: {singleImageY}")
plt.show()
