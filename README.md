# 🌟 **Team 1: Deep Learning** - Klasifikasi Suara Nyamuk Berbasis CNN untuk Inovasi Pengendalian Hama dan Penyakit

Proyek ini dikembangkan oleh **Team-1** dari kelas **Deep Learning 2024**. Tujuan utamanya adalah mengklasifikasikan suara kepakan sayap nyamuk berdasarkan spesies menggunakan model **Convolutional Neural Network (CNN)** untuk mendukung inovasi dalam pengendalian hama dan penyakit di wilayah tropis, khususnya di Indonesia.

---

## 🎯 **Fokus Proyek**

### **Spesies yang Diklasifikasi:**
- 🧟🏼‍♂️ **_Aedes aegypti_**: Vektor utama penyakit demam berdarah dengue (DBD).
- 🧟🏼‍♂️ **_Anopheles stephensi_**: Vektor utama malaria di wilayah tropis.
- 🧟🏼‍♂️ **_Culex pipiens_**: Vektor penyakit filariasis.

### **Dukungan terhadap Target Pemerintah:**
- **Eliminasi malaria dan filariasis** pada tahun 2030.
- Penurunan **insiden DBD** menjadi di bawah **49 kasus per 100.000 jiwa**.

### **Manfaat Proyek:**
- Meningkatkan deteksi dini spesies nyamuk berbahaya.
- Memberikan solusi berbasis teknologi untuk pengendalian hama dan penyakit tropis.

![🧟 Gambar Nyamuk](https://github.com/sains-data/Klasifikasi-Suara-Nyamuk-Berbasis-CNN-untuk-Inovasi-Pengendalian-Hama-dan-Penyakit/blob/main/Deploy/Logo0.jpg)

---

## 👥 **Anggota Kelompok**
1. **Ignatius Krisna Issaputra** (121140037)  
2. **Ardoni Yeriko Rifana Gultom** (121140141)  
3. **Rika Ajeng Finatih** (121450036)  
4. **M. Gilang Martiansyah** (121450056)  
5. **Sasa Rahma Lia** (121450119)  
6. **Nazwa Nabilla** (121450122)  

---

## 🚀 **Tujuan Proyek**
- Mengembangkan **sistem klasifikasi suara nyamuk otomatis** untuk mendeteksi spesies nyamuk.
- Mendukung **inovasi pengendalian hama** dan penyakit tropis berbasis teknologi AI.

---

## 📂 **Dataset**

Dataset yang digunakan meliputi:
- **Rekaman audio nyamuk** dalam format **.wav**
- **Label spesies** dalam file **.csv**

**🔗 [Download Dataset](https://drive.google.com/drive/folders/109Spn_kf2DCFK1Xqb1f9K2w70kUPVaAj?usp=sharing)**

### **Pengolahan Data:**
1. Audio difilter untuk menghilangkan noise.
2. Fitur diekstraksi menggunakan **MFCC** dan **Mel Spectrogram**.
3. Data di-augmentasi untuk meningkatkan generalisasi model.

---

## 🛠️ **Teknologi yang Digunakan**
| Teknologi          | Deskripsi                                                                 |
|--------------------|---------------------------------------------------------------------------|
| <img src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg" width="80"> | **Python**: Bahasa pemrograman utama untuk pemrosesan data dan pengembangan model. |
| <img src="https://media.wired.com/photos/5927105acfe0d93c474323d7/master/pass/google-tensor-flow-logo-black-S.jpg" width="80"> | **TensorFlow/Keras**: Framework untuk membangun dan melatih model deep learning. |
| <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcReiDgm71NRUOVsiA_rTGi8lsIZmO1rlYt4cw&s" width="80"> | **Librosa**: Pustaka untuk analisis dan ekstraksi fitur audio (MFCC, Mel Spectrogram). |
| <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Adobe_Audition_CC_icon_%282020%29.svg/800px-Adobe_Audition_CC_icon_%282020%29.svg.png" width="80"> | **Adobe Audition**: Untuk preprocessing audio dan pembersihan noise. |
| <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT1IS9rNAuZkFawNTS7W3dgsNcuOjNfh9imKQ&s" width="80"> | **Streamlit**: Framework untuk membuat aplikasi web interaktif. |
| <img src="https://matplotlib.org/stable/_static/logo2.svg" width="80"> | **Matplotlib & Seaborn**: Untuk visualisasi data dan evaluasi kinerja model. |

---

## 🧬 Model CNN _(Convolutional Neural Network)_

### **Arsitektur Model**
- Model dirancang untuk memproses spektrum audio dari suara nyamuk.
- Menggunakan **2-3 layer convolusi** diikuti oleh **max-pooling** dan **fully connected layer**.
- Optimasi dilakukan menggunakan **Adam Optimizer**.

### **Teknik yang Digunakan:**
1. **Augmentasi Data:** Menghasilkan variasi data melalui pitch shifting dan time stretching.
2. **Regularisasi:** Menggunakan dropout untuk mencegah overfitting.
3. **Evaluasi:** Metrik utama meliputi akurasi, precision, recall, dan F1-score.

---

## 📊 Metodologi
1. **Preprocessing Audio:**
   - Rekaman suara diolah untuk menghasilkan fitur MFCC dan Mel Spectrogram.
2. **Pelatihan Model:**
   - CNN dilatih dengan parameter yang dioptimalkan untuk klasifikasi spesies.
3. **Evaluasi Model:**
   - Performa model dievaluasi menggunakan data validasi dan tes.

---

## 🏆 **Hasil yang Diharapkan**
- **Akurasi klasifikasi ≥ 75%** untuk semua spesies nyamuk.
- **Waktu prediksi sistem ≤ 1 detik** per suara.
- **Sensitivitas ≥ 80%** untuk deteksi spesies nyamuk yang relevan.

---

## 📢 Flowchart Proses
![Deskripsi Gambar](https://github.com/sains-data/Klasifikasi-Suara-Nyamuk-Berbasis-CNN-untuk-Inovasi-Pengendalian-Hama-dan-Penyakit/blob/main/flowchart%20.png)

---

## 👨‍💻 Cara Menjalankan Proyek

1. **Clone repositori**:
   ```bash
   git clone https://github.com/sains-data/Klasifikasi-Suara-Nyamuk-Berbasis-CNN-untuk-Inovasi-Pengendalian-Hama-dan-Penyakit.git
   ```
2. **Masuk ke direktori proyek** yang baru di-clone:
   ```bash
   cd Klasifikasi-Suara-Nyamuk-Berbasis-CNN-untuk-Inovasi-Pengendalian-Hama-dan-Penyakit
   ```
3. **Instal dependensi**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Jalankan notebook atau script**:
   ```bash
   python main.py
   ```

---

## 👥 **Kontributor**
| Nama                           | Github                                                 |
|--------------------------------|--------------------------------------------------------|
| Ignatius Krisna Issaputra      | [Github](https://github.com/inExcelsis1710)            |
| Ardoni Yeriko Rifana Gultom    | [Github](https://github.com/gultom20)                  |
| Rika Ajeng Finatih             | [Github](https://github.com/rika623)                   |
| M. Gilang Martiansyah          | [Github](https://github.com/mgilang56)                 |
| Sasa Rahma Lia                 | [Github](https://github.com/sasarahmalia)              |
| Nazwa Nabilla                  | [Github](https://github.com/nazwanabila)               |

---

## 📢 Kontak
Jika ada pertanyaan, silakan hubungi:

- **Email:**
  - ignatius.121140037@student.itera.ac.id
  - ardoni.121140141@student.itera.ac.id
  - rika.121450036@student.itera.ac.id
  - mgilang.121450056@student.itera.ac.id
  - sasa.121450119@student.itera.ac.id
  - nazwa.121450122@student.itera.ac.id

---

## 🙏 Ucapan Terima Kasih

Kami ingin mengucapkan terima kasih yang sebesar-besarnya kepada:

### **1. Dosen Pembimbing**
- **Bapak Ardika Satria, S.Si., M.Si.**: Atas bimbingan dan arahan yang sangat berharga selama pengerjaan proyek ini.

### **2. Dosen Mata Kuliah**
- **Bapak Christyan Tamaro Nadeak, M.Si**: Atas ilmu yang diberikan dalam mata kuliah Deep Learning.
- **Ibu Ade Lailani, M.Si**: Atas kontribusi dalam pengajaran konsep-konsep dasar yang mendukung proyek ini.

### **3. Anggota Kelompok**
- **Ardoni Yeriko Rifana Gultom**: Mengembangkan model CNN dan berkontribusi dalam preprocessing data.
- **M. Gilang Martiansyah**: Membuat aplikasi prediksi menggunakan Streamlit dan evaluasi model.
- **Rika Ajeng Finatih**: Memimpin proyek dan mendokumentasikan laporan.
- **Ignatius Krisna Issaputra**: Ekstraksi fitur audio dan pengembangan model CNN.
- **Sasa Rahma Lia**: Penyusunan dokumentasi teknis.
- **Nazwa Nabila**: Pengujian dan pelaporan sistem.

---

## 🔗 **Tautan Penting**

- **Notion Kelompok 1**: [Notion](https://aquamarine-dove-b45.notion.site/Team-1-Proyek-Tugas-Besar-Deep-Learning-133607a60e95805294dada205aea761d)  
- **Demo Aplikasi Streamlit**: [Coba Aplikasi](https://mosquitoclassify1.streamlit.app/) _(disarankan memakai mode gelap untuk pengalaman terbaik)_  
- **Train Model CNN**: [Download Model](https://drive.google.com/file/d/1rbfhPOQLBKxyRvrSUS5jpHjjVBGgCKqx/view?usp=drive_link)  
- **Train History JSON**: [Download History](https://drive.google.com/file/d/1tl_NtfvabLha3-hrwYIaQmPu3hrxYgYv/view?usp=drive_link)  

---

## 🎥 **Demo Video**
**In Progress**: [Tonton Video](https://www.youtube.com/watch?v=XXXXXXX)

---

