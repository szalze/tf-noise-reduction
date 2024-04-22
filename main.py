import tensorflow as tf
import os
from tensorflow.keras import layers, models

os.environ['TF_ENApipBLE_ONEDNN_OPTS'] = '0'

# Függvény a .wav fájlok betöltéséhez és feldolgozásához
def load_and_preprocess_wav(file_path):
    # Betöltés
    audio_binary = tf.io.read_file(file_path)
    # Feldolgozás, például normalizálás, jellemző kinyerése stb.
    # Ide illeszthetsz zajszűrési vagy jellemző kinyerési logikát
    # Egy példa: normalizálás
    audio, _ = tf.audio.decode_wav(audio_binary)
    audio = tf.squeeze(audio, axis=-1)  # Több csatornás audio esetén
    normalized_audio = audio / tf.math.reduce_max(audio)
    return normalized_audio

# Függvény a .wav fájlok listájának előállításához
def get_wav_files_list(directory):
    wav_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.wav')]
    return wav_files

# Függvény a TensorFlow Record-ok létrehozásához
def create_tf_record(wav_files, output_file):
    with tf.io.TFRecordWriter(output_file) as writer:
        for wav_file in wav_files:
            # .wav fájl betöltése és feldolgozása
            audio_data = load_and_preprocess_wav(wav_file)
            # TensorFlow Record létrehozása
            example = tf.train.Example(features=tf.train.Features(feature={
                'audio': tf.train.Feature(float_list=tf.train.FloatList(value=audio_data.numpy()))
            }))
            # TFRecord írása
            writer.write(example.SerializeToString())

# A .wav fájlok elérési útvonala
wav_directory = 'noisy_files'
# TFRecord kimeneti fájl elérési útvonala
output_tfrecord_file = 'output.tfrecord'

# .wav fájlok betöltése és feldolgozása
wav_files = get_wav_files_list(wav_directory)
# TensorFlow Record-ok létrehozása
create_tf_record(wav_files, output_tfrecord_file)

