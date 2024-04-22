import tensorflow as tf

# Szükséges függvények importálása
stft = tf.signal.stft
inverse_stft = tf.signal.inverse_stft
sample_rate = 44100

def trim_audio_to_seconds(audio_waveform, target_seconds, sample_rate):
    target_length = int(target_seconds * sample_rate)
    current_length = tf.shape(audio_waveform)[0]

    if current_length > target_length:
        trimmed_waveform = audio_waveform[:target_length]
    else:
        # Ha a jelenlegi hossz rövidebb, mint a célszám, egyszerűen másoljuk azt és vegyük az üres helyet.
        trimmed_waveform = tf.concat([audio_waveform, tf.zeros(target_length - current_length)], axis=0)

    return trimmed_waveform

def spectral_subtraction(noisy_waveform, cleaner_waveform, cut_interval_sec):
    # Mindkét hangfájlt 6 másodperces hosszúságra vágjuk
    trimmed_noisy_waveform = trim_audio_to_seconds(noisy_waveform, cut_interval_sec, sample_rate)
    trimmed_cleaner_waveform = trim_audio_to_seconds(cleaner_waveform, cut_interval_sec, sample_rate)

    noisy_stft = stft(trimmed_noisy_waveform, frame_length=256, frame_step=128)
    cleaner_stft = stft(trimmed_cleaner_waveform, frame_length=256, frame_step=128)

    noise_spec = tf.abs(noisy_stft) - tf.abs(cleaner_stft)
    noise_spec = tf.maximum(noise_spec, 0)

    denoised_spec = tf.abs(noisy_stft) - noise_spec
    denoised_waveform = inverse_stft(denoised_spec, frame_length=256, frame_step=128)

    return denoised_waveform

# Betöltjük a referencia zajmintát
cleaner_sample = "noise_files/waccleaner_1.wav"
cleaner_waveform, _ = tf.audio.decode_wav(tf.io.read_file(cleaner_sample))

# Betöltjük a zajos hangfájlt
noisy_audio_file = "data_audio/noisy_speech.wav"
noisy_waveform, _ = tf.audio.decode_wav(tf.io.read_file(noisy_audio_file))

# Zajszűrés a zajos hangfájlon
denoised_audio = spectral_subtraction(noisy_waveform, cleaner_waveform, 6)

# Eredmény kiírása
output_file = "denoised_audio.wav"
tf.audio.encode_wav(denoised_audio, sample_rate).numpy()
tf.io.write_file(output_file, denoised_audio)

print("Zajszűrés kész! Az eredmény a", output_file, "fájlban található.")
