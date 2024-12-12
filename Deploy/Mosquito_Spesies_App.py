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
from streamlit_lottie import st_lottie
import requests
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

def load_training_history(file_path=HISTORY_FILE):
    """Load the training history from a JSON file."""
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except Exception as e:
        st.error(f"Error loading training history: {e}")
        return None

def load_and_preprocess_file(audio_file):
    """Load and preprocess the audio file."""
    try:
        # Membaca file audio yang di-upload sebagai byte stream
        audio_bytes = audio_file.read()
        audio_buffer = io.BytesIO(audio_bytes)

        # Load audio using librosa from byte stream
        y, sr = librosa.load(audio_buffer, sr=None)

        # Extract Mel-spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # Extract MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

        # Resize features
        target_shape = (180, 180)
        mel_spectrogram_resized = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        mfcc_resized = resize(np.expand_dims(mfcc, axis=-1), target_shape)

        # Combine features
        X_test = np.concatenate((mel_spectrogram_resized, mfcc_resized), axis=-1)
        return np.expand_dims(X_test, axis=0)  # Add batch dimension
    except Exception as e:
        print(f"Error during audio preprocessing: {e}")
        return None

def model_prediction(X_test, model):
    """Perform prediction using the model."""
    try:
        prediction = model.predict(X_test)
        return np.argmax(prediction, axis=1)[0]  # Return predicted class index
    except Exception as e:
        st.error(f"Error during model prediction: {e}")
        return None

def show_prediction_result(audio_file, model):
    """Display the prediction result along with the spectrogram."""
    X_test = load_and_preprocess_file(audio_file)
    if X_test is not None:
        result_index = model_prediction(X_test, model)
        if result_index is not None:
            labels = ["Aedes Aegypti", "Anopheles Stephensi", "Culex Pipiens"]
            st.markdown(f"**Predicted Species:** {labels[result_index]}")

            # Visualisasi Spektrogram
            audio_file.seek(0)  # Reset file pointer to the start
            y, sr = librosa.load(audio_file, sr=None)
            mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
            mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

            # Plot Mel-Spectrogram
            plt.figure(figsize=(10, 6))
            librosa.display.specshow(mel_spectrogram_db, x_axis='time', y_axis='mel', sr=sr)
            plt.colorbar(format='%+2.0f dB')
            plt.title('Mel-Spectrogram')
            st.pyplot(plt)

        else:
            st.error("Model failed to provide a prediction.")
    else:
        st.error("Failed to process the audio file.")

def plot_training_history():
    """Plot the training history of the model."""
    history = load_training_history()
    if history is not None:
        # Plot accuracy
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        st.pyplot(plt)

# UI Styling Functions
def add_bg_from_url():
    """Add background from a URL."""
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url('https://jenis.net/wp-content/uploads/2020/06/jenis-nyamuk-e1591437296119-768x456.jpg');
            background-size: cover;
            background-position: top center;
            color: white;
        }
        h1, h2, h3, h4, h5 {
            color: white;
            text-align: center;
        }
        .footer {
            position: fixed;
            bottom: 0;
            width: 100%;
            background-color: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 10px;
            text-align: center;
        }
        .center-content {
            display: flex;
            justify-content: center;
            flex-direction: column;
            align-items: center;
        }
        .social-icons {
            position: fixed;
            bottom: 10px;
            left: 10px;
            display: flex;
            gap: 15px;
        }
        .social-icons img {
            width: 40px;
            height: 40px;
            cursor: pointer;
        }
        .custom-button {
            background-color: rgba(0, 0, 0, 0.7);
            color: white;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            font-size: 16px;
            margin-top: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def add_header_logo():
    """Add header logo and title."""
    st.markdown(
        """
        <div class="center-content">
            <div style="display: flex; justify-content: center; gap: 20px; margin-bottom: 20px;">
                <div style="border: 2px solid white; border-radius: 10px; padding: 5px;">
                    <img src="https://github.com/sains-data/Klasifikasi-Suara-Nyamuk-Berbasis-CNN-untuk-Inovasi-Pengendalian-Hama-dan-Penyakit/blob/main/Deploy/Logo1.png?raw=true" alt="Logo 1" width="100" height="100">
                </div>
                <div style="border: 2px solid white; border-radius: 10px; padding: 5px;">
                    <img src="https://github.com/sains-data/Klasifikasi-Suara-Nyamuk-Berbasis-CNN-untuk-Inovasi-Pengendalian-Hama-dan-Penyakit/blob/main/Deploy/Logo2.png?raw=true" alt="Logo 2" width="100" height="100">
                </div>
                <div style="border: 2px solid white; border-radius: 10px; padding: 5px;">
                    <img src="https://github.com/sains-data/Klasifikasi-Suara-Nyamuk-Berbasis-CNN-untuk-Inovasi-Pengendalian-Hama-dan-Penyakit/blob/main/Deploy/Logo3.png?raw=true" alt="Logo 3" width="100" height="100">
                </div>
            </div>
            <h1>Klasifikasi Suara Nyamuk Berdasarkan Spesies Berbasis CNN untuk Inovasi Pengendalian Hama dan Penyakit</h1>
        </div>
        <div class="social-icons">
            <a href="https://github.com/mgilang56/TugasBesarDeeplearningKel1" target="_blank">
                <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a0/GitHub_Mark.svg/1024px-GitHub_Mark.svg.png" alt="GitHub" />
            </a>
            <a href="https://www.linkedin.com/in/mgilang56/" target="_blank">
                <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/9d/LinkedIn_logo_initials.png/512px-LinkedIn_logo_initials.png" alt="LinkedIn" />
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

def add_user_guide():
    """Add user guide for the application."""
    st.markdown("""
    <div class="center-content">
        <h2>Panduan Penggunaan:</h2>
        <ul style="font-size: 18px;">
            <li>1. Unggah file audio dari nyamuk yang ingin diklasifikasikan.</li>
            <li>2. Klik tombol untuk memprediksi spesies nyamuk berdasarkan suara.</li>
            <li>3. Hasil prediksi akan ditampilkan beserta visualisasi Mel-spectrogram.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Main Streamlit App
def main():
    """Main function for the Streamlit app."""
    add_bg_from_url()  # Add background
    add_header_logo()  # Add header logo and title
    add_user_guide()   # Add user guide
    
    st.markdown("### Upload File Audio Nyamuk:")
    
    # Upload file audio
    audio_file = st.file_uploader("Pilih file audio (.wav, .mp3)", type=["wav", "mp3"])
    
    if audio_file is not None:
        model = load_model()  # Load the model
        show_prediction_result(audio_file, model)  # Show the prediction result
        plot_training_history()  # Show the training history plot

# Run the app
if __name__ == "__main__":
    main()
