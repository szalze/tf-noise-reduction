import librosa
import numpy as np
import soundfile as sf


# Hangfájl MFCC-ekké alakítása
def wav_to_mfcc(input_file, output_file):
    # Hangfájl beolvasása
    audio, sr = librosa.load(input_file, sr=None)

    # MFCC-k kinyerése
    mfccs = librosa.feature.mfcc(y=audio, sr=sr)

    # MFCC-k mentése
    np.save(output_file, mfccs)


# MFCC-ekből hangfájl visszaállítása
def mfcc_to_wav(input_file, output_file, sr=22050):
    # MFCC-k betöltése
    mfccs = np.load(input_file)

    # Hang visszaállítása a MFCC-kból
    audio = librosa.feature.inverse.mfcc_to_audio(mfccs)

    # Hang mentése
    sf.write(output_file, audio, sr)


# Hangfájl MFCC-ekké alakítása és visszaállítása
def convert_wav(input_file, output_file):
    # Hangfájl MFCC-ekké alakítása
    mfcc_file = output_file + ".npy"
    wav_to_mfcc(input_file, mfcc_file)

    # MFCC-ekből hangfájl visszaállítása
    mfcc_to_wav(mfcc_file, output_file)


# Teszt
input_file = r"C:\drone dataset\Lajcsi\stereo_split\stereo_split_002.wav"
output_file = r"C:\Users\szalz\PycharmProjects\projekt\mffc_output\output1.wav"
convert_wav(input_file, output_file)
