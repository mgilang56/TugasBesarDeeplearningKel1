import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from tensorflow.image import resize
import io
import os
import gdown
import json


# URL Google Drive
model_url = "https://drive.google.com/uc?id=1rbfhPOQLBKxyRvrSUS5jpHjjVBGgCKqx"
history_url = "https://drive.google.com/uc?id=1tl_NtfvabLha3-hrwYIaQmPu3hrxYgYv"

# Nama file setelah diunduh
model_file = "trained_model.keras"
history_file = "training_history.json"

# Fungsi untuk mengunduh file dari Google Drive
def download_file_from_google_drive(url, output):
    if not os.path.exists(output):
        with st.spinner(f"Downloading {output}..."):
            gdown.download(url, output, quiet=False)

# Fungsi untuk memuat model
def load_model():
    download_file_from_google_drive(model_url, model_file)
    model = tf.keras.models.load_model(model_file)
    return model

# Fungsi untuk memuat riwayat pelatihan
def load_training_history(file_path="training_history.json"):
    try:
        with open(file_path, "r") as file:
            history = json.load(file)
        return history
    except Exception as e:
        st.error(f"Error loading training history: {e}")
        return None

# Fungsi untuk memuat dan memproses file audio
def load_and_preprocess_file(audio_file):
    # Memuat file audio menggunakan librosa
    y, sr = librosa.load(audio_file, sr=None)
    
    # Melakukan ekstraksi fitur mel-spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    # Melakukan ekstraksi fitur MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Resize fitur agar sesuai dengan ukuran input model
    target_shape = (180, 180)  # Menyesuaikan dengan ukuran input model Anda
    
    # Melakukan resize pada mel-spectrogram dan MFCC
    mel_spectrogram_resized = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
    mfcc_resized = resize(np.expand_dims(mfcc, axis=-1), target_shape)
    
    # Menggabungkan fitur mel-spectrogram dan MFCC
    X_test = np.concatenate((mel_spectrogram_resized, mfcc_resized), axis=-1)
    
    # Menambahkan dimensi tambahan untuk batch
    X_test = np.expand_dims(X_test, axis=0)
    
    return X_test

# Fungsi untuk melakukan prediksi menggunakan model
def model_prediction(X_test, model):
    try:
        # Melakukan prediksi
        prediction = model.predict(X_test)
        result_index = np.argmax(prediction, axis=1)[0]  # Menentukan indeks kelas yang diprediksi
        return result_index
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model atau prediksi: {e}")
        return None

# Fungsi untuk menampilkan hasil prediksi
def show_prediction_result(audio_file, model):
    X_test = load_and_preprocess_file(audio_file)
    
    # Menampilkan bentuk data setelah preprocessing untuk debugging
    st.write("Shape of preprocessed data:", X_test.shape)
    
    if X_test is not None:
        result_index = model_prediction(X_test, model)
        if result_index is not None:
            # Daftar label kelas yang diprediksi
            label = ["Aedes Aegypti", "Anopheles Stephensi", "Culex Pipiens"]
            st.markdown(f"Predicted Species: {label[result_index]}")
        else:
            st.error("Model gagal memberikan prediksi.")
    else:
        st.error("Gagal memproses file audio.")

# Fungsi utama untuk menjalankan aplikasi Streamlit
def main():
    st.title("Prediksi Spesies Nyamuk Berdasarkan Suara")
    
    # Mengunduh model dan riwayat pelatihan
    model = load_model()
    
    # Upload file audio
    audio_file = st.file_uploader("Pilih file audio untuk diprediksi", type=["wav", "mp3"])
    
    if audio_file is not None:
        # Menampilkan file yang dipilih
        st.audio(audio_file, format="audio/wav")
        
        # Menampilkan hasil prediksi
        show_prediction_result(audio_file, model)

# Menjalankan aplikasi Streamlit
if __name__ == "__main__":
    main()
