# Emotion Recognition System dengan YOLOv12 dan EfficientNet BiLSTM

## Deskripsi Sistem
Sistem pengenalan emosi real-time menggunakan YOLOv12 untuk deteksi wajah dan EfficientNet BiLSTM untuk klasifikasi emosi. Sistem ini mendukung:
- Upload gambar
- Capture dari webcam
- Rekam video
- Deteksi real-time
- Chat bot responsif berdasarkan emosi

## Masalah yang Ditemukan dan Solusi

### 1. **Error Variabel Tidak Terdefinisi** ✅ DIPERBAIKI
**Masalah:** Variabel `current_frame_detected_names` digunakan tanpa didefinisikan.
**Solusi:** Menambahkan `current_frame_detected_names = set()` di awal fungsi.

### 2. **Path Model Salah** ✅ DIPERBAIKI
**Masalah:** Kode mencari `emotion_efficientnet_bilstm.pth` padahal file bernama `emotion_efficientnet_bilstm_trained.pth`.
**Solusi:** Mengubah path ke nama file yang benar.

### 3. **Masalah Threading** ✅ DIPERBAIKI
**Masalah:** Race condition pada cache global dengan threading mode.
**Solusi:** 
- Mengubah ke `async_mode="eventlet"`
- Menambahkan thread-safe cache dengan `threading.Lock()`
- Menambahkan `eventlet` ke requirements

### 4. **Error Handling Model Loading** ✅ DIPERBAIKI
**Masalah:** Sistem crash jika model tidak bisa dimuat.
**Solusi:** 
- Menambahkan error handling yang lebih baik
- Sistem tetap berjalan dengan model default jika ada masalah
- Menampilkan warning yang informatif

### 5. **Memory Management** ✅ DIPERBAIKI
**Masalah:** Cache dan buffer tidak dikelola dengan baik.
**Solusi:** 
- Menambahkan thread-safe cache
- Membersihkan buffer untuk wajah yang tidak terdeteksi
- Membatasi ukuran cache

## Instalasi

1. **Clone repository:**
```bash
git clone <repository-url>
cd EvisionWeb
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Pastikan model files ada:**
- `models/yolov12s.pt`
- `models/emotion_efficientnet_bilstm_trained.pth`

4. **Jalankan aplikasi:**
```bash
python app.py
```

## Struktur File
```
EvisionWeb/
├── app.py                          # Main Flask application
├── config.py                       # Configuration settings
├── requirements.txt                # Python dependencies
├── models/                         # Model files
│   ├── yolov12s.pt
│   └── emotion_efficientnet_bilstm_trained.pth
├── src/                           # Source code
│   ├── emotion_detector.py        # Core detection logic
│   ├── efficientnet_bilstm_model.py
│   └── model_loader.py
├── templates/                     # HTML templates
├── static/                        # CSS, JS, images
├── known_faces/                   # Known face images
└── uploads/                       # Uploaded files
```

## Fitur Utama

### 1. **Deteksi Wajah dengan YOLOv12**
- Deteksi wajah real-time
- Bounding box dengan confidence score
- Support multiple faces

### 2. **Pengenalan Wajah**
- Face recognition menggunakan face_recognition library
- Cache untuk performa yang lebih baik
- Support untuk multiple known faces

### 3. **Deteksi Emosi dengan EfficientNet BiLSTM**
- Model temporal untuk sequence analysis
- 7 emosi: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprised
- Confidence threshold untuk filtering

### 4. **Web Interface**
- Upload gambar
- Capture dari webcam
- Rekam dan proses video
- Real-time detection
- Chat bot responsif

### 5. **Chat Bot dengan Gemini API**
- Respons berdasarkan emosi terdeteksi
- Support bahasa Indonesia dan Inggris
- Tone yang disesuaikan dengan emosi

## Troubleshooting

### Model Loading Issues
Jika ada masalah loading model:
1. Pastikan file model ada di folder `models/`
2. Check log untuk error messages
3. Sistem akan tetap berjalan dengan fallback mode

### Performance Issues
1. Kurangi confidence threshold untuk deteksi lebih cepat
2. Gunakan GPU jika tersedia
3. Monitor memory usage

### WebSocket Issues
1. Pastikan eventlet terinstall: `pip install eventlet`
2. Check browser console untuk error
3. Restart server jika ada masalah

## API Endpoints

- `GET /` - Home page
- `POST /upload` - Upload dan proses gambar
- `POST /capture` - Capture dari webcam
- `POST /record` - Rekam dan proses video
- `GET /realtime` - Real-time detection page
- `WebSocket /frame` - Real-time frame processing
- `WebSocket /chat_message` - Chat functionality

## Dependencies

### Core Dependencies
- Flask 3.1.0
- Flask-SocketIO 5.5.1
- OpenCV 4.11.0
- PyTorch 2.6.0
- Ultralytics 8.3.82
- face_recognition

### Additional Dependencies
- eventlet 0.35.2 (untuk async mode)
- langdetect 1.0.9 (untuk deteksi bahasa)
- requests 2.32.3 (untuk Gemini API)

## Contributing

1. Fork repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## License

MIT License - lihat file LICENSE untuk detail.

## Support

Jika ada masalah atau pertanyaan, silakan buat issue di repository ini.

## **Fitur Baru yang Ditambahkan:**

### 1. **Multiple Foto per Orang** ✅
Sistem sekarang mendukung multiple foto per orang untuk pengenalan wajah yang lebih akurat:

```
known_faces/
├── nesya/
│   ├── nesya1.jpg
│   ├── nesya2.png
│   └── nesya3.jpg
├── oji/
│   ├── oji1.jpg
│   ├── oji2.png
│   └── oji3.jpg
├── prabowo/
│   ├── prabowo1.jpg
│   └── prabowo2.png
└── fadel.jpg  # Foto langsung di root folder
```

### 2. **Chart Monitoring Emosi** ✅
- **Emotion Distribution Chart**: Doughnut chart menampilkan distribusi emosi
- **Confidence Over Time Chart**: Line chart menampilkan confidence level seiring waktu
- **Emotion Log**: Log real-time dengan timestamp dan warna sesuai emosi

### 3. **Logging Emosi yang Lebih Detail** ✅
- Log setiap deteksi emosi dengan confidence level
- Informasi semua probabilitas emosi
- Debugging yang lebih mudah

### 4. **Perbaikan Responsivitas Emosi** ✅
- Mengurangi requirement frame dari 5 menjadi 3 untuk deteksi lebih cepat
- Padding frame untuk konsistensi
- Fallback yang lebih baik

## **Cara Menggunakan Multiple Foto:**

1. **Buat folder untuk setiap orang** di `known_faces/`
2. **Masukkan beberapa foto** orang tersebut dalam folder
3. **Restart aplikasi** untuk memuat ulang wajah
4. **Sistem akan otomatis** mengenali semua foto dalam folder

## **Cara Menggunakan Chart:**

1. **Buka halaman Real-time** (`/realtime`)
2. **Klik "Start Detection"** untuk memulai
3. **Lihat chart** yang update secara real-time:
   - **Emotion Distribution**: Distribusi emosi yang terdeteksi
   - **Confidence Over Time**: Grafik confidence level
   - **Emotion Log**: Log detail setiap deteksi

## **API Endpoint Baru:**

- `GET /api/emotion-data`: Mendapatkan data emosi untuk analytics
  - Returns: recent emotions, distribution, total records
