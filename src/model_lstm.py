import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
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
model_lstm = Sequential([
    LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1)),
    Dense(1)
])

# Compile the model
model_lstm.compile(optimizer='adam', loss='mean_squared_error')

# Define callbacks (optional)
checkpoint = ModelCheckpoint('./results/model_lstm.keras', monitor='val_loss',
                             save_best_only=True, mode='min', verbose=1)

# Train the model
model_lstm.fit(X_train, y_train,
              epochs=50, batch_size=32,
              validation_data=(X_test, y_test),
              callbacks=[checkpoint])
