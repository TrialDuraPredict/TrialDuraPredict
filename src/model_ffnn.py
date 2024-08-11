import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint

# import dataset
with open('./results/X_train.pkl', 'rb') as file:
    X_train = pickle.load(file)

with open('./results/y_train.pkl', 'rb') as file:
    y_train = pickle.load(file)
    
with open('./results/X_test.pkl', 'rb') as file:
    X_test = pickle.load(file)

with open('./results/y_test.pkl', 'rb') as file:
    y_test = pickle.load(file)

# Initialize the model
model_ffnn = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])

# Compile the model
model_ffnn.compile(optimizer='adam', loss='mean_squared_error')

# Define callbacks (optional but recommended for saving the best model)
checkpoint = ModelCheckpoint('./results/model_ffnn.keras', monitor='val_loss',
                             save_best_only=True, mode='min', verbose=1)

# Train the model
model_ffnn.fit(X_train, y_train,
               epochs=50, batch_size=32,
               validation_data=(X_test, y_test),
               callbacks=[checkpoint])
