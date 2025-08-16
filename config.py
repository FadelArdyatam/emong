
# Project Emong
# File: config.py
import os

# Jalur ke direktori utama proyek
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Jalur ke model YOLO
MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov12s.pt")

# Jalur ke folder penyimpanan uploads (gambar, video)
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Jalur ke file video sementara
TEMP_VIDEO_PATH = os.path.join(UPLOADS_DIR, "temp_video.mp4")

# Jalur ke file video hasil rekaman webcam
RECORDED_VIDEO_PATH = os.path.join(UPLOADS_DIR, "recorded_video.mp4")

# Jalur ke gambar hasil tangkapan webcam
CAPTURED_IMAGE_PATH = os.path.join(UPLOADS_DIR, "captured_image.jpg")

# API key untuk Gemini (ganti dengan API key sebenarnya)
GEMINI_API_KEY = "AIzaSyAEGNoQjDSQQOCd8tCf2JrDJQNt6_6QFwE"  # Ganti dengan API key asli
