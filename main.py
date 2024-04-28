import tensorflow as tf
import librosa
import numpy as np
import soundfile as sf

# Hangfájlok betöltése és előfeldolgozása
def load_and_preprocess_audio(file_path, sample_rate=48000):
    audio, _ = librosa.load(file_path, sr=sample_rate)
    audio_tensor = tf.convert_to_tensor(audio)

    normalized_audio = tf.keras.layers.Rescaling(scale=1. / 127.5, offset=-1)(audio_tensor)

    # MFCC tenzor kiterjesztése
    #audio_tensor_reshaped = tf.expand_dims(normalized_audio, axis=0)  # Batch dimenzió hozzáadása
    # audio_tensor_reshaped = tf.expand_dims(audio_tensor_reshaped, axis=-1)  # Utolsó dimenzió hozzáadása
    audio_tensor_reshaped = tf.convert_to_tensor(normalized_audio, dtype=tf.float32)
    audio_tensor_reshaped = tf.stack([tf.ones((94, 30), dtype=tf.float32)])
    return audio_tensor_reshaped

# Hangszegmensek szűrése a betanított modell súlyainak betöltésekor
def filter_audio(model_path, audio_tensor):
    try:
        # Teljes modell betöltése a .keras fájlból
        model = tf.keras.models.load_model(model_path)
        print("Keras model loaded successfully.")
    except (OSError, ValueError):
        print("Failed to load the Keras model.")
        return None

    # Hangszegmensek szűrése a betanított modell súlyainak betöltésekor
    filtered_audio = model(audio_tensor)
    return filtered_audio

# Hangfájl betöltése és előfeldolgozása
audio_path = r'C:\drone dataset\szurendo.wav'
audio_tensor = load_and_preprocess_audio(audio_path)

# Súlyfájl betöltése és hangszegmensek szűrése
model_path = r'saved_model\cnn_rnn_model.keras'
filtered_audio = filter_audio(model_path, audio_tensor)

# Tenzor visszaalakítása hanganyaggá
filtered_audio_np = filtered_audio.numpy()
#filtered_audio_np = np.squeeze(filtered_audio_np)  # Eltávolítja a felesleges dimenziót
print(filtered_audio_np)

# Szűrt hangfájl mentése
output_file_path = r'output\output_file.wav'
sf.write(output_file_path, filtered_audio_np, 48000)
