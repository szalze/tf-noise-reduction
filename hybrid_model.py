import glob
import tensorflow as tf
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
import os
from tensorflow.keras.layers import LSTM, Rescaling, Conv1D, MaxPooling1D, Dense, TimeDistributed, Dropout
import matplotlib.pyplot as plt

# Define hyperparameters
target_length = 48000  # Target audio length
n_mfcc = 30  # Number of MFCCs to extract
batch_size = 64 # Batch size
epochs = 10  # Number of epochs
learning_rate = 0.0001  # Learning rate

# Data Preprocessing function using MFCCs
def prepare_dataset(audio_file_paths, clean_file_paths):
    audio_data = []
    labels = []
    for file_path, clean_path in zip(audio_file_paths, clean_file_paths):
        try:
            # Load noisy and clean audio files
            audio, _ = librosa.load(file_path)
            clean_audio, _ = librosa.load(clean_path)

            # Pad or truncate audio files to target length
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)), 'constant')
                clean_audio = np.pad(clean_audio, (0, target_length - len(clean_audio)), 'constant')
            else:
                audio = audio[:target_length]
                clean_audio = clean_audio[:target_length]

            # Process audio with MFCC
            mfccs_noise = librosa.feature.mfcc(y=audio, sr=48000, n_mfcc=n_mfcc)
            audio_data.append(mfccs_noise.T)

            # Label: clean audio
            mfcss_clean = librosa.feature.mfcc(y=clean_audio, sr=48000, n_mfcc=n_mfcc)
            labels.append(mfcss_clean.T)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")


    X = tf.cast(np.array(audio_data), dtype=tf.float32)
    y = tf.cast(np.array(labels), dtype=tf.float32)
    """
    X = tf.reshape(audio_data, (-1, len(audio_data), 1))
    y = tf.reshape(labels, (-1, len(labels), 1))
    """


    print(X.shape, y.shape)
    return X, y


# CNN-RNN Model Architecture
def create_cnn_rnn_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    # Rescaling for normalization
    normalized_audio = Rescaling(scale=1./127.5, offset=-1)(inputs)

    # 1st Conv1D layer
    conv1 = Conv1D(filters=16, kernel_size=3, activation='relu', padding='same')(normalized_audio)

    # Dropout
    dp1 = Dropout(0.2)(conv1)

    # Max pooling
    pool1 = MaxPooling1D(pool_size=2, padding='same')(dp1)

    # Dropout
    dp2 = Dropout(0.2)(pool1)

    #LSTM
    lstm = LSTM(64, return_sequences=True)(dp2)

    #Attention
    attention = tf.keras.layers.Attention()([lstm, lstm])


    # Output
    outputs = TimeDistributed(Dense(1, activation='sigmoid'))(attention)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# Load audio files
drone_directory = r'drone_noise'
clean_directory = r'clean_noise'

# Check and prepare dataset
if os.path.exists(drone_directory) and os.path.isdir(drone_directory):
    drone_files = glob.glob(os.path.join(drone_directory, '*.wav'))
    clean_files = glob.glob(os.path.join(clean_directory, '*.wav'))
    if not drone_files or not clean_files:
        print("No audio files found.")
else:
    print("The specified directories do not exist or are not directories.")

# Prepare dataset using MFCCs
X, y = prepare_dataset(drone_files, clean_files)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X.numpy(), y.numpy(), test_size=0.2, random_state=42)

# Create and compile CNN-RNN model
model = create_cnn_rnn_model(X_train.shape[1:])
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))

# Save the model
save_path = r'saved_model\cnn_rnn_model'
model.save(save_path + '.keras')
print(f"Model saved to {save_path}")

# Save the weights
save_path = r'saved_model\cnn_rnn_model'
model.save_weights(save_path + '.weights.h5')
print(f"Model weights saved to {save_path}")

# Evaluate model performance on validation set
print("Evaluating model on validation set...")
val_loss, val_acc = model.evaluate(X_val, y_val)
print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
