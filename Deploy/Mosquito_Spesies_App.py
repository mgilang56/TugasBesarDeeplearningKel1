import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from tensorflow.image import resize
import io
import os
import gdown
import json
import matplotlib.pyplot as plt


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

# Menambahkan background gambar dari URL
def add_bg_from_url():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url('https://asset.kompas.com/crops/Uoby6be9TIeMzC18327oT1MCjlI=/13x0:500x325/1200x800/data/photo/2020/03/12/5e69cae0eb1d1.jpg');
            background-size: cover;
            background-position: top center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Menambahkan desain UI
def add_custom_style():
    st.markdown(
        """
        <style>
        body {
            background-size: cover;
            color: #FFFFFF;  /* Warna putih untuk teks */
            font-family: 'Roboto', sans-serif;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            text-align: center;
        }

        h1 {
            font-size: 2.5em;
            font-weight: bold;
            color: #FFFFFF;  /* Warna putih */
            text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.6); /* Menambahkan bayangan pada teks */
        }

        h3 {
            font-size: 1.5em;
            color: #FFFFFF;  /* Warna putih */
            text-shadow: 1px 1px 6px rgba(0, 0, 0, 0.6); /* Bayangan pada teks */
        }

        .upload-form {
            background-color: rgba(255, 255, 255, 0.2); /* Background transparan putih */
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
            display: flex;
            flex-direction: column;
            align-items: center;
            margin-top: 20px;
        }

        input[type="file"] {
            margin: 10px 0;
            padding: 10px;
            border: none;
            border-radius: 5px;
            background-color: #333; /* Background hitam */
            color: white;
            font-size: 16px;
            cursor: pointer;
        }

        input[type="submit"] {
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            background-color: #3b3d6b;
            color: white;
            font-size: 16px;
            cursor: pointer;
            transition: background-color 0.3s;
        }

        input[type="submit"]:hover {
            background-color: #5f5f91;
        }

        img {
            margin-top: 20px;
            max-width: 100%;
            border-radius: 15px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        }

        .footer {
            margin-top: 20px;
            font-size: 1rem;
            color: #FFFFFF;  /* Warna putih untuk footer */
            font-weight: bold;
            position: fixed;
            bottom: 10px;
            width: 100%;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Fungsi untuk memuat dan memproses file audio
def load_and_preprocess_file(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    
    # Ekstraksi mel-spectrogram dan MFCC
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Resize fitur agar sesuai dengan ukuran input model
    target_shape = (180, 180)
    mel_spectrogram_resized = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
    mfcc_resized = resize(np.expand_dims(mfcc, axis=-1), target_shape)
    
    X_test = np.concatenate((mel_spectrogram_resized, mfcc_resized), axis=-1)
    X_test = np.expand_dims(X_test, axis=0)
    
    return X_test

# Fungsi untuk melakukan prediksi menggunakan model
def model_prediction(X_test, model):
    try:
        prediction = model.predict(X_test)
        result_index = np.argmax(prediction, axis=1)[0]  # Menentukan indeks kelas yang diprediksi
        return result_index
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model atau prediksi: {e}")
        return None

# Fungsi untuk menampilkan hasil prediksi
def show_prediction_result(audio_file, model):
    X_test = load_and_preprocess_file(audio_file)
    
    if X_test is not None:
        result_index = model_prediction(X_test, model)
        if result_index is not None:
            label = ["Aedes Aegypti", "Anopheles Stephensi", "Culex Pipiens"]
            st.markdown(f"Predicted Species: {label[result_index]}")
        else:
            st.error("Model gagal memberikan prediksi.")
    else:
        st.error("Gagal memproses file audio.")

# Fungsi utama untuk menjalankan aplikasi Streamlit
def main():
    add_bg_from_url()  # Menambahkan latar belakang
    add_custom_style()  # Menambahkan style custom

    st.markdown("""
        <div>
            <img src="https://github.com/sains-data/Klasifikasi-Suara-Nyamuk-Berbasis-CNN-untuk-Inovasi-Pengendalian-Hama-dan-Penyakit/blob/main/Deploy/Logo1.png?raw=true" alt="Logo Nyamuk 1" width="65" height="65">
            <img src="https://github.com/sains-data/Klasifikasi-Suara-Nyamuk-Berbasis-CNN-untuk-Inovasi-Pengendalian-Hama-dan-Penyakit/blob/main/Deploy/Logo2.png?raw=true" alt="Logo Nyamuk 2" width="65" height="65">
            <img src="https://github.com/sains-data/Klasifikasi-Suara-Nyamuk-Berbasis-CNN-untuk-Inovasi-Pengendalian-Hama-dan-Penyakit/blob/main/Deploy/Logo3.png?raw=true" alt="Logo Nyamuk 3" width="65" height="65">
        </div>
        <h1>Klasifikasi Suara Nyamuk Berdasarkan Spesiesnya Berbasis CNN</h1>
        <h3>Upload file suara nyamuk untuk memprediksi spesiesnya</h3>
    """, unsafe_allow_html=True)

    # Mengunduh model
    model = load_model()

    # Formulir upload file audio
    st.title("Upload Audio Suara Nyamuk")
    test_wav = st.file_uploader("Pilih file audio...", type=["wav"])

    if test_wav is not None:
        audio_bytes = test_wav.read()

        if st.button("Predict"):
            with st.spinner("Predicting..."):
                X_test = load_and_preprocess_file(io.BytesIO(audio_bytes))
                if X_test is not None:
                    result_index = model_prediction(X_test, model)
                    if result_index is not None:
                        label = ["Aedes Aegypti", "Anopheles Stephensi", "Culex Pipiens"]
                        st.markdown(f"Predicted Species: {label[result_index]}")

if __name__ == "__main__":
    main()
