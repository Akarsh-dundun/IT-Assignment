import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
# Load MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize the data
X_train = X_train / 255.0
X_test = X_test / 255.0

# One-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Create a simple neural network model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(90, activation='relu'),
    Dense(60, activation='relu'),
    Dense(30, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# Save the model
model.save('lab2_model.h5')
