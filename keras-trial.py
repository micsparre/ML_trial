from tensorflow import keras
import numpy as np
from tensorflow.keras import layers
from keras.layers.experimental.preprocessing import Normalization
from matplotlib import pyplot as plt

# model and data parameters
num_classes = 10
input_shape = (28, 28, 1)

# load and split the mnist digit dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


# add a normalization layer to normalize the data
norm_layer = Normalization()
norm_layer.adapt(x_train)
print('x_train shape: ' + str(x_train.shape))
print('x_test shape: ' + str(x_test.shape))
print('y_train shape: ' + str(y_train.shape))
print('y_test shape: ' + str(y_test.shape))

# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# build the model
model = keras.Sequential(
    [
        # norm_layer,
        # keras.Input(shape=input_shape),
        keras.layers.InputLayer(input_shape=input_shape),
        norm_layer,
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

# train the model
batch_size = 128
epochs = 15
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

predicted = model.predict(x_test[:50], batch_size = 10)
for i in range(5):
    fig, ax = plt.subplots(2, 5, figsize=(15, 6))
    for axi, j in zip(ax.flat, range(10)):
        axi.set(xticks=[], yticks=[])
        axi.imshow(x_test[i * 10 + j])
        axi.set_title(f'Prediction: {predicted[i * 10 + j].argmax()}. Actual: {y_test[i * 10 + j].argmax()}.')
    plt.show(block=True)
