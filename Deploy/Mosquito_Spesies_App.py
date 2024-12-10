import streamlit as st
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from skimage.transform import resize

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
def model_prediction(X_test):
    try:
        # Load model
        model = tf.keras.models.load_model('model_culex.h5')  # Pastikan nama model benar
        # Melakukan prediksi
        prediction = model.predict(X_test)
        result_index = np.argmax(prediction, axis=1)[0]  # Menentukan indeks kelas yang diprediksi
        return result_index
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model atau prediksi: {e}")
        return None

# Fungsi untuk menampilkan hasil prediksi
def show_prediction_result(audio_file):
    X_test = load_and_preprocess_file(audio_file)
    
    # Menampilkan bentuk data setelah preprocessing untuk debugging
    st.write("Shape of preprocessed data:", X_test.shape)
    
    if X_test is not None:
        result_index = model_prediction(X_test)
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
    
    # Upload file audio
    audio_file = st.file_uploader("Pilih file audio untuk diprediksi", type=["wav", "mp3"])
    
    if audio_file is not None:
        # Menampilkan file yang dipilih
        st.audio(audio_file, format="audio/wav")
        
        # Menampilkan hasil prediksi
        show_prediction_result(audio_file)

# Menjalankan aplikasi Streamlit
if __name__ == "__main__":
    main()
