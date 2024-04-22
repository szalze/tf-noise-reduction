import os

import numpy
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split


# Hangfájlok betöltésére szolgáló függvény
def load_audio_files(file_paths, sample_rate=44100):
    audio_data = []
    for file_path in file_paths:
        audio, _ = librosa.load(file_path, sr=sample_rate, duration=10.0)  # Minden hangfájl 1 másodperces lesz
        audio_data.append(audio)
    return audio_data


# Adatkészlet előkészítésére szolgáló függvény
def prepare_dataset(drone_paths, sample_rate=44100):
    drone_audio = load_audio_files(drone_paths, sample_rate)

    X = np.array(drone_audio)
    y = np.ones(len(drone_audio))  # 1 a drónzajhoz

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


# RNN modell létrehozására szolgáló függvény
def create_rnn_model(input_shape):
    from tensorflow.keras.layers import Input

    # Bemeneti réteg definiálása
    model_input = Input(shape=input_shape[1:])

    # RNN modell definiálása
    model = tf.keras.Sequential([
        model_input,
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation='sigmoid'))
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Modell tanítására és mentésére szolgáló függvény
def train_and_save_model(X_train, y_train, save_path):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    model = create_rnn_model(X_train.shape)
    model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.2, callbacks=[early_stopping])
    model.save_weights(save_path)


# Elérési útvonal a drónzaj hangfájlokhoz
drone_directory = r'C:\Users\szalz\PycharmProjects\projekt\dataset'

# Ellenőrizzük, hogy a megadott elérési útvonal valós és tartalmaz fájlokat
if os.path.exists(drone_directory) and os.path.isdir(drone_directory):
    drone_files = []
    for i in range(1, 89):
        file_name = f'ch_right_{i:03d}.wav'
        file_path = os.path.join(drone_directory, file_name)
        if os.path.exists(file_path):
            drone_files.append(file_path)
else:
    print("A megadott elérési útvonal nem létezik vagy nem egy könyvtár.")

# Adatkészlet előkészítése
X_train, X_test, y_train, y_test = prepare_dataset(drone_files)

# Tanítás és mentés
save_path = 'rnn_model_weights.h5'
print(X_train, y_train)

try:
    train_and_save_model(X_train, y_train, save_path)
    # Kiértékelés
    rnn_model = create_rnn_model(X_train.shape)
    rnn_model.load_weights(save_path)
    loss, accuracy = rnn_model.evaluate(X_test, y_test)
    print("Test Accuracy:", accuracy)

except ValueError as v_err:
    print(X_train)
    print(y_train)
    print(v_err)
    print(v_err.args)
    print(str(v_err))
    pass
finally:
    print("SCRIPT FINISHED")
