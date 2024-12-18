import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import librosa.display
from tensorflow.image import resize
import io
import os
import gdown
import matplotlib.pyplot as plt

# Google Drive URLs
MODEL_URL = "https://drive.google.com/uc?id=1rbfhPOQLBKxyRvrSUS5jpHjjVBGgCKqx"
MODEL_FILE = "trained_model.keras"

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

        # Mel-spectrogram extraction
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        resized = resize(np.expand_dims(mel_spectrogram, axis=-1), (180, 180))
        return np.expand_dims(resized, axis=0)
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

def model_prediction(X_test, model):
    """Perform prediction using the trained model."""
    try:
        prediction = model.predict(X_test)
        return np.argmax(prediction, axis=1)[0]
    except Exception as e:
        st.error(f"Error during model prediction: {e}")
        return None

# UI Styling Functions
def add_background_and_css():
    """Add background and social media styling."""
    st.markdown("""
    <style>
    .stApp {
        background-image: url("https://github.com/mgilang56/TugasBesarDeeplearningKel1/blob/main/Deploy/backgroundcoba2.jpg?raw=true");
        background-size: cover;
        background-position: top center;
    }
    .social-icons {
        position: fixed;
        bottom: 20px;
        left: 20px;
        display: flex;
        gap: 10px;
    }
    .social-icons img {
        width: 40px;
        height: 40px;
        cursor: pointer;
    }
    </style>
    """, unsafe_allow_html=True)

def add_header_logo():
    """Add header logos and title."""
    st.markdown("""
    <div style="text-align: center;">
        <div style="display: flex; justify-content: center; gap: 15px; margin-bottom: 10px;">
            <img src="https://github.com/sains-data/Klasifikasi-Suara-Nyamuk-Berbasis-CNN-untuk-Inovasi-Pengendalian-Hama-dan-Penyakit/blob/main/Deploy/Logo1.png?raw=true" width="80">
            <img src="https://github.com/sains-data/Klasifikasi-Suara-Nyamuk-Berbasis-CNN-untuk-Inovasi-Pengendalian-Hama-dan-Penyakit/blob/main/Deploy/Logo2.png?raw=true" width="80">
            <img src="https://github.com/sains-data/Klasifikasi-Suara-Nyamuk-Berbasis-CNN-untuk-Inovasi-Pengendalian-Hama-dan-Penyakit/blob/main/Deploy/Logo3.png?raw=true" width="80">
        </div>
        <h1 style="color: white; font-size: 30px; margin: 0;">Klasifikasi Suara Nyamuk Berbasis CNN</h1>
        <h2 style="color: white; font-size: 22px; margin-top: 0;">untuk Inovasi Pengendalian Hama dan Penyakit</h2>
    </div>
    """, unsafe_allow_html=True)

def add_social_icons():
    """Add clickable social media icons."""
    st.markdown("""
    <div class="social-icons">
        <a href="https://github.com/mgilang56/TugasBesarDeeplearningKel1" target="_blank">
            <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub">
        </a>
        <a href="https://wa.me/6285157725574" target="_blank">
            <img src="https://upload.wikimedia.org/wikipedia/commons/6/6b/WhatsApp.svg" alt="WhatsApp">
        </a>
        <a href="https://instagram.com" target="_blank">
            <img src="https://upload.wikimedia.org/wikipedia/commons/a/a5/Instagram_icon.png" alt="Instagram">
        </a>
    </div>
    """, unsafe_allow_html=True)

# Main Function
def main():
    add_background_and_css()
    add_header_logo()
    add_social_icons()

    st.markdown("### Upload File Audio Nyamuk:")
    audio_file = st.file_uploader("Pilih file audio (.wav, .mp3)", type=["wav", "mp3"])

    if audio_file:
        model = load_model()
        X_test = load_and_preprocess_file(audio_file)
        if X_test is not None:
            labels = ["Aedes Aegypti", "Anopheles Stephensi", "Culex Pipiens"]
            result = model_prediction(X_test, model)
            if result is not None:
                st.success(f"Predicted Species: **{labels[result]}**")
                st.image(
                    f"https://github.com/mgilang56/TugasBesarDeeplearningKel1/blob/main/Deploy/{labels[result].replace(' ', '%20')}.png?raw=true",
                    use_column_width=True
                )
        else:
            st.error("Failed to process the uploaded audio file.")

if __name__ == "__main__":
    main()
