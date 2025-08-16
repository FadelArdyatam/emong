import cv2
import numpy as np
import face_recognition
import os
import torch # Added for PyTorch operations
from torchvision import transforms # Added for image preprocessing

# --- Helper Functions (can be moved to a utils.py if desired) ---

# Define transformations for the EfficientNet BiLSTM model
# These should match the transformations used during training
emotion_transform = transforms.Compose([
    transforms.ToPILImage(), # Convert OpenCV image to PIL Image
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet normalization
])

# Define the sequence length for the BiLSTM model
SEQUENCE_LENGTH = 5 # This must match the sequence_length used during training
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
    emotion_model, # This is now the EfficientNet BiLSTM model
    image,
    known_face_encodings,
    known_face_names,
    recent_face_cache, # New parameter for caching
    face_sequence_buffers, # New parameter to store sequences for BiLSTM
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

        current_frame_detected_names.add(name) # Add detected name to set for current frame

        # 2b. Deteksi Emosi pada Wajah menggunakan EfficientNet BiLSTM
        # Use the recognized name as a key for the sequence buffer
        if name not in face_sequence_buffers:
            face_sequence_buffers[name] = []

        # Preprocess the face image for the emotion model
        processed_face_tensor = emotion_transform(face_image)

        # Add the processed face to the sequence buffer
        face_sequence_buffers[name].append(processed_face_tensor)

        # Keep the buffer size limited to SEQUENCE_LENGTH
        if len(face_sequence_buffers[name]) > SEQUENCE_LENGTH:
            face_sequence_buffers[name].pop(0) # Remove the oldest frame

        emotion_label = "Neutral" # Default if not enough frames or no clear prediction
        emotion_conf = 0.0

        # Only make a prediction if we have enough frames in the sequence
        if len(face_sequence_buffers[name]) == SEQUENCE_LENGTH:
            # Stack the sequence to create a batch for the model
            # Add batch dimension: (1, SEQUENCE_LENGTH, C, H, W)
            sequence_batch = torch.stack(face_sequence_buffers[name]).unsqueeze(0)

            # Move to CPU for inference (assuming model is on CPU)
            # If you moved the model to GPU in app.py, you'll need to move this to GPU too
            # sequence_batch = sequence_batch.to(device) # if using GPU

            with torch.no_grad(): # No need to calculate gradients for inference
                emotion_output = emotion_model(sequence_batch)
                probabilities = torch.softmax(emotion_output, dim=1)
                max_conf, predicted_idx = torch.max(probabilities, 1)

                # Map predicted_idx to emotion label
                # This list MUST match the order of classes used during your model training
                emotion_labels_list = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprised"]
                
                if predicted_idx.item() < len(emotion_labels_list):
                    emotion_label = emotion_labels_list[predicted_idx.item()]
                    emotion_conf = max_conf.item()

                if emotion_conf < emotion_confidence_threshold:
                    emotion_label = "Neutral" # Fallback if confidence is too low

        detections.append([name, emotion_label, emotion_conf, (x1, y1, x2, y2)])

    # Clean up buffers for faces not detected in the current frame
    # This is a simple way to remove stale entries. More robust tracking would be better.
    keys_to_delete = [name for name in face_sequence_buffers if name not in current_frame_detected_names]
    for key in keys_to_delete:
        del face_sequence_buffers[key]

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
