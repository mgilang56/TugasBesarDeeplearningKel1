import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import librosa.display
from tensorflow.image import resize
import io
import os
import gdown
import json
import matplotlib.pyplot as plt
from datetime import datetime
import pytz
import soundfile as sf

# Google Drive URLs
MODEL_URL = "https://drive.google.com/uc?id=1rbfhPOQLBKxyRvrSUS5jpHjjVBGgCKqx"
HISTORY_URL = "https://drive.google.com/uc?id=1tl_NtfvabLha3-hrwYIaQmPu3hrxYgYv"

# File Names
MODEL_FILE = "trained_model.keras"
HISTORY_FILE = "training_history.json"

# Utility Functions
def download_file(url, output):
    """Download file from Google Drive if not already downloaded."""
    if not os.path.exists(output):
        with st.spinner(f"Downloading {output}..."):
            gdown.download(url, output, quiet=False)

def load_model():
    """Load the trained model."""
    download_file(MODEL_URL, MODEL_FILE)
    return tf.keras.models.load_model(MODEL_FILE)

def load_and_preprocess_file(audio_file):
    """Load and preprocess the audio file."""
    try:
        audio_bytes = audio_file.read()
        audio_buffer = io.BytesIO(audio_bytes)
        y, sr = librosa.load(audio_buffer, sr=None)

        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        target_shape = (180, 180)

        mel_resized = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        mfcc_resized = resize(np.expand_dims(mfcc, axis=-1), target_shape)

        X_test = np.concatenate((mel_resized, mfcc_resized), axis=-1)
        return np.expand_dims(X_test, axis=0)
    except Exception as e:
        st.error(f"Error during audio preprocessing: {e}")
        return None

def model_prediction(X_test, model):
    try:
        prediction = model.predict(X_test)
        return np.argmax(prediction, axis=1)[0]
    except Exception as e:
        st.error(f"Error during model prediction: {e}")
        return None

def show_prediction_result(audio_file, model):
    X_test = load_and_preprocess_file(audio_file)
    if X_test is not None:
        result_index = model_prediction(X_test, model)
        if result_index is not None:
            labels = ["Aedes Aegypti", "Anopheles Stephensi", "Culex Pipiens"]
            st.success(f"Predicted Species: {labels[result_index]}")

def add_custom_styles():
    st.markdown("""
    <style>
    .stApp {
        background-image: url("https://github.com/mgilang56/TugasBesarDeeplearningKel1/blob/main/Deploy/backgroundcoba2.jpg?raw=true");
        background-size: cover;
        background-position: center;
    }
    .logo-container img {
        border: 3px solid white;
        border-radius: 15px;
    }
    .guide-box {
        background: rgba(255, 255, 255, 0.9);
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        margin: auto;
    }
    </style>
    ", unsafe_allow_html=True)

def add_header_logo():
    st.markdown("""
    <div style="text-align: center;">
        <div class="logo-container" style="display: flex; justify-content: center; gap: 15px;">
            <img src="https://github.com/sains-data/Klasifikasi-Suara-Nyamuk-Berbasis-CNN-untuk-Inovasi-Pengendalian-Hama-dan-Penyakit/blob/main/Deploy/Logo1.png?raw=true" width="80" height="80">
            <img src="https://github.com/sains-data/Klasifikasi-Suara-Nyamuk-Berbasis-CNN-untuk-Inovasi-Pengendalian-Hama-dan-Penyakit/blob/main/Deploy/Logo2.png?raw=true" width="80" height="80">
            <img src="https://github.com/sains-data/Klasifikasi-Suara-Nyamuk-Berbasis-CNN-untuk-Inovasi-Pengendalian-Hama-dan-Penyakit/blob/main/Deploy/Logo3.png?raw=true" width="80" height="80">
        </div>
        <h1 style="color: white; font-size: 30px; margin-top: 10px;">Klasifikasi Suara Nyamuk Berdasarkan Spesies Berbasis CNN</h1>
        <h2 style="color: white; font-size: 20px; margin-top: -10px;">untuk Inovasi Pengendalian Hama dan Penyakit</h2>
    </div>
    ", unsafe_allow_html=True)

def add_user_guide():
    st.markdown("""
    <div class="guide-box">
        <h3 style="text-align: center; color: black;">Panduan Penggunaan:</h3>
        <ul style="font-size: 18px;">
            <li>Unggah file audio dari nyamuk yang ingin diklasifikasikan.</li>
            <li>Klik tombol untuk memprediksi spesies nyamuk berdasarkan suara.</li>
            <li>Hasil prediksi akan ditampilkan beserta visualisasi Mel-spectrogram.</li>
        </ul>
    </div>
    ", unsafe_allow_html=True)

def main():
    add_custom_styles()
    add_header_logo()
    add_user_guide()

    st.markdown("### Upload File Audio Nyamuk:")
    audio_file = st.file_uploader("Pilih file audio (.wav, .mp3)", type=["wav", "mp3"])

    if audio_file is not None:
        model = load_model()
        show_prediction_result(audio_file, model)

if __name__ == "__main__":
    main()
