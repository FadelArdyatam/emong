import cv2
import numpy as np
import face_recognition
import os

# --- Helper Functions (can be moved to a utils.py if desired) ---

def get_image_quality(image):
    """
    Menganalisis kualitas gambar berdasarkan kecerahan, kontras, dan blur.
    """
    if image is None or image.size == 0:
        return 0, 0, 0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    contrast = np.std(gray)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    return brightness, contrast, blur

def preprocess_for_emotion(face_image):
    """
    Mempersiapkan gambar wajah untuk deteksi emosi (contoh: normalisasi).
    """
    # Normalisasi histogram untuk meningkatkan kontras
    gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    equalized_face = cv2.equalizeHist(gray_face)
    # Konversi kembali ke BGR karena model YOLO mengharapkan 3 channel
    return cv2.cvtColor(equalized_face, cv2.COLOR_GRAY2BGR)

def load_known_faces(known_faces_dir='known_faces'):
    """
    Memuat wajah yang dikenal dari direktori untuk pengenalan.
    """
    known_face_encodings = []
    known_face_names = []
    if not os.path.exists(known_faces_dir):
        print(f"Warning: Known faces directory not found at '{known_faces_dir}'")
        return known_face_encodings, known_face_names

    for filename in os.listdir(known_faces_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(known_faces_dir, filename)
            name = os.path.splitext(filename)[0]
            try:
                image = face_recognition.load_image_file(path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    known_face_encodings.append(encodings[0])
                    known_face_names.append(name)
            except Exception as e:
                print(f"Error loading known face {filename}: {e}")
    return known_face_encodings, known_face_names

MAX_CACHE_SIZE = 10 # Maximum number of recent faces to cache

def get_emotion_colors():
    """
    Mendefinisikan warna BGR untuk setiap emosi untuk visualisasi.
    """
    return {
        'Anger': (0, 0, 255), 'Contempt': (0, 128, 255), 'Disgust': (0, 255, 0),
        'Fear': (128, 0, 128), 'Happy': (0, 255, 255), 'Neutral': (255, 255, 255),
        'Sad': (255, 0, 0), 'Surprised': (255, 165, 0), 'Unknown': (128, 128, 128)
    }

# --- Core Detection Logic ---

def detect_emotions_and_recognize_faces(
    face_detector_model,
    emotion_model,
    image,
    known_face_encodings,
    known_face_names,
    recent_face_cache, # New parameter for caching
    face_confidence_threshold=0.4, # Keyakinan minimum untuk deteksi wajah
    emotion_confidence_threshold=0.3 # Keyakinan minimum untuk deteksi emosi
):
    """
    Arsitektur baru: Deteksi wajah (YOLO), lalu pengenalan & deteksi emosi.
    """
    # 1. Deteksi Wajah Cepat dengan YOLO
    face_results = face_detector_model(image, verbose=False) # verbose=False untuk output bersih
    if not face_results or not hasattr(face_results[0], 'boxes'):
        return []

    face_boxes = face_results[0].boxes
    detections = []

    # 2. Proses Setiap Wajah yang Ditemukan
    for box in face_boxes:
        conf = float(box.conf)
        if conf < face_confidence_threshold:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Crop wajah dari gambar asli
        # Tambahkan sedikit padding untuk konteks yang lebih baik
        padding = 10
        face_image = image[max(0, y1-padding):min(image.shape[0], y2+padding), 
                             max(0, x1-padding):min(image.shape[1], x2+padding)]

        if face_image.size == 0:
            continue

        # 2a. Pengenalan Wajah (Identifikasi Nama)
        # Perlu gambar dalam format RGB untuk face_recognition
        rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_face)
        name = "Unknown"
        if face_encodings:
            # Ambil encoding pertama (asumsi satu wajah per crop)
            face_encoding = face_encodings[0]

            # --- Caching Logic ---
            found_in_cache = False
            for i, (cached_encoding, cached_name) in enumerate(recent_face_cache):
                # Use a slightly higher tolerance for cache matching to be more forgiving
                if face_recognition.compare_faces([cached_encoding], face_encoding, tolerance=0.5)[0]:
                    name = cached_name
                    found_in_cache = True
                    # Move to end to act as LRU (Least Recently Used)
                    recent_face_cache.append(recent_face_cache.pop(i))
                    break
            
            if not found_in_cache:
                # Compare with known faces if not found in cache
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]
                
                # Add/Update to cache
                # Remove if already exists to update its position (LRU)
                recent_face_cache[:] = [(enc, n) for enc, n in recent_face_cache if not face_recognition.compare_faces([enc], face_encoding, tolerance=0.5)[0]]
                recent_face_cache.append((face_encoding, name))
                # Limit cache size
                if len(recent_face_cache) > MAX_CACHE_SIZE:
                    recent_face_cache.pop(0) # Remove oldest (first element)

        # 2b. Deteksi Emosi pada Wajah
        preprocessed_face = preprocess_for_emotion(face_image)
        emotion_results = emotion_model(preprocessed_face, verbose=False)
        
        emotion_label = "Neutral" # Default jika tidak ada emosi terdeteksi
        emotion_conf = 0.0

        if emotion_results and hasattr(emotion_results[0], 'boxes'):
            # Cari emosi dengan confidence tertinggi dari hasil deteksi
            highest_conf = -1
            best_emotion = "Neutral"
            for emotion_box in emotion_results[0].boxes:
                if emotion_box.conf > highest_conf:
                    highest_conf = float(emotion_box.conf)
                    cls = int(emotion_box.cls)
                    best_emotion = emotion_model.names[cls]
            
            if highest_conf > emotion_confidence_threshold:
                emotion_label = best_emotion
                emotion_conf = highest_conf

        detections.append([name, emotion_label, emotion_conf, (x1, y1, x2, y2)])

    return detections

# --- Drawing Function ---

def draw_bounding_box(image, detections, emotion_colors_bgr):
    """
    Menggambar kotak pembatas dengan nama dan emosi pada gambar asli.
    """
    for name, emotion, confidence, (x1, y1, x2, y2) in detections:
        color = emotion_colors_bgr.get(emotion, (128, 128, 128))
        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        conf_percent = confidence * 100
        label = f"{name} - {emotion}: {conf_percent:.0f}%"
        
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        label_y = y1 - 10 if y1 - 10 > 10 else y2 + text_height + 10
        
        # Latar belakang untuk label agar mudah dibaca
        cv2.rectangle(image, (x1, label_y - text_height - 5), (x1 + text_width, label_y + 5), color, -1)
        cv2.putText(image, label, (x1, label_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        
    return image
