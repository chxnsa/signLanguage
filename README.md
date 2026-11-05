# Klasifikasi ASL Alphabet dengan CNN dan Transfer Learning

Tugas individu mata kuliah Deep Learning, Universitas Hasanuddin.
Nama: [nama lengkap] | NIM: [NIM]

## Tujuan
Membandingkan dua arsitektur deep learning, yaitu CNN yang dibangun dari nol
dan Transfer Learning (MobileNetV2), untuk mengklasifikasikan gambar
bahasa isyarat ASL.

## Dataset
- Sumber: [ASL Alphabet (Kaggle, grassknoted)](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- 29 kelas: A-Z, del, nothing, space
- Subset yang dipakai: 500 gambar per kelas (total 14.500 gambar), diacak dengan seed 42
- Ukuran gambar diubah menjadi 128x128
- Pembagian data stratified 70/15/15: 10.150 train, 2.175 validasi, 2.175 test

## Arsitektur
**Model 1: CNN from scratch**
- Rescaling, augmentasi (RandomRotation 0.05, RandomZoom 0.1)
- 3 blok Conv2D (32, 64, 128 filter) + MaxPooling
- Flatten, Dropout 0.5, Dense 128, Dense 29 (softmax)
- Total parameter: 4,29 juta

**Model 2: Transfer Learning (MobileNetV2, pretrained ImageNet)**
- Tahap 1 (frozen): base dibekukan, hanya head (GlobalAveragePooling, Dropout 0.3, Dense 29) yang dilatih, Adam lr 1e-3
- Tahap 2 (fine-tuning): 30 layer terakhir dibuka (BatchNorm tetap beku), Adam lr 1e-5
- Augmentasi sama dengan CNN agar perbandingan adil

Early stopping (monitor val_loss, restore best weights) dipakai di semua tahap.

## Hasil
| Model | Test Accuracy | Waktu training |
|---|---|---|
| CNN from scratch | 95,77% | 2,6 menit |
| MobileNetV2 (frozen) | 94,39% | 2,2 menit |
| MobileNetV2 + fine-tuning | 96,32% | 3,6 menit (2,2 + 1,4) |

Macro F1 model terbaik: 0,96.

## Analisis
- Fine-tuning memberi hasil terbaik. Versi frozen masih underfit karena
  MobileNetV2 dilatih pada foto benda umum, sedangkan tugas ini
  membedakan bentuk jari yang mirip.
- Selisih fine-tuning dengan CNN hanya 0,55 poin dari satu kali percobaan,
  jadi keduanya sebaiknya dianggap sebanding.
- Kelas del, nothing, dan space terklasifikasi sempurna.
- Error terbanyak pada huruf S (precision 0,78), X (F1 0,88), serta pasangan
  M-N, V-K, dan W-V. Huruf-huruf ini berbentuk tangan menggenggam atau
  berjari mirip.

## Keterbatasan
Dataset direkam dalam kondisi seragam (orang dan background serupa) dan data
test berasal dari distribusi yang sama dengan data train, sehingga akurasi
belum tentu berlaku pada foto dunia nyata. Hasil juga berasal dari satu kali
run tanpa pengulangan dengan seed berbeda.

## Cara menjalankan
1. Buka notebook `asl_classification.ipynb` di Kaggle atau Colab (aktifkan GPU)
2. Jalankan sel dari atas ke bawah (dataset diunduh otomatis lewat kagglehub)

## Teknologi
Python, TensorFlow/Keras, scikit-learn, matplotlib, seaborn