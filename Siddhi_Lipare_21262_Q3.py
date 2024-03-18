import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

# Define the number of neurons per layer
neurons_per_layer = [512, 256, 128]

# Define the activation functions for each layer
activation_functions = ['relu', 'relu', 'softmax']

# Define the model
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))  # Assuming input images are 28x28 pixels
for i in range(len(neurons_per_layer)):
    model.add(Dense(neurons_per_layer[i], activation=activation_functions[i]))

# Compile the model
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()
