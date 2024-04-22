import tensorflow as tf
import soundfile as sf


# Adatok betöltése és előkészítése
sample_rate = 44100
cleaner_audio_file = 'noise_files/waccleaner_1.wav'  # Tisztított hanghullám adatok betöltése
noisy_audio_file = 'data_audio/noisy_speech.wav'    # Zajos hanghullám adatok betöltése


cleaner_waveform, _ = tf.audio.decode_wav(tf.io.read_file(cleaner_audio_file))
noisy_waveform, _ = tf.audio.decode_wav(tf.io.read_file(noisy_audio_file))

# Trimmelés
#min_length = min(len(cleaner_waveform), len(noisy_waveform))
min_length = len(cleaner_waveform)

cleaner_waveform = cleaner_waveform[:min_length]
noisy_waveform = noisy_waveform[:min_length]

# Tanító adatok előkészítése
X_train = tf.reshape(noisy_waveform, (-1, len(noisy_waveform), 1))
y_train = tf.reshape(cleaner_waveform, (-1, len(cleaner_waveform), 1))

# Definiáljuk az SNR függvényt
def signal_to_noise_ratio(y_true, y_pred):
    numerator = tf.reduce_sum(tf.square(y_true), axis=-1)
    denominator = tf.reduce_sum(tf.square(y_true - y_pred), axis=-1)
    snr = 10 * tf.math.log(numerator / denominator) / tf.math.log(10.0)
    return snr

# Definiáljuk az SDR függvényt
def signal_to_distortion_ratio(y_true, y_pred):
    numerator = tf.reduce_sum(tf.square(y_true), axis=-1)
    denominator = tf.reduce_sum(tf.square(y_true - y_pred), axis=-1)
    sdr = 10 * tf.math.log(numerator / denominator) / tf.math.log(10.0)
    return sdr


def train_model(X_train, y_train):
    # Modell létrehozása
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(32, 5, activation='relu', padding='same', input_shape=X_train.shape[1:]),
        tf.keras.layers.Conv1D(32, 5, activation='relu', padding='same'),
        tf.keras.layers.Conv1D(1, 5, padding='same')
    ])

    # Modell összeállítása és tanítása
    #model.compile(optimizer='adam', loss='sdr')
    #model.compile(optimizer='adam', loss='mse')
    #model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError(), metrics=[signal_to_noise_ratio])
    #model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=[signal_to_noise_ratio])
    #model.fit(X_train, y_train, epochs=10, batch_size=32)
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    return model


# Modell tanítása
trained_model = train_model(X_train, y_train)

# Zajszűrés a tanult modellel
denoised_waveform = trained_model.predict(X_train)
denoised_waveform = denoised_waveform.squeeze(axis=-1)

# Eredmény kiírása
output_file = "denoised_audio.wav"

print("Denoised waveform shape:", denoised_waveform.shape)
print("Denoised waveform dtype:", denoised_waveform.dtype)

# Hangfájl létrehozása
sf.write(output_file, denoised_waveform[0], sample_rate)

print("Zajszűrés kész! Az eredmény a", output_file, "fájlban található.")
