import torch
from ultralytics import YOLO
import os

# --- Konfigurasi Path ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_PATH = os.path.join(BASE_DIR, 'models', 'yolov8s.pt')
EMOTION_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'emotion_efficientnet_bilstm_trained.pth')

print("--- Memulai Pengecekan File Model ---")

# --- 1. Cek Model YOLO ---
print(f"\n1. Mencoba memuat model YOLO dari: {YOLO_PATH}")
if not os.path.exists(YOLO_PATH):
    print("   [GAGAL] File tidak ditemukan.")
else:
    try:
        model = YOLO(YOLO_PATH)
        print("   [BERHASIL] Model YOLO berhasil dimuat.")
    except Exception as e:
        print(f"   [GAGAL] Terjadi error saat memuat model YOLO: {e}")
        print("   --- TRACEBACK ERROR YOLO ---")
        import traceback
        traceback.print_exc()
        print("   ---------------------------")


# --- 2. Cek Model Emosi ---
print(f"\n2. Mencoba memuat model Emosi dari: {EMOTION_MODEL_PATH}")
if not os.path.exists(EMOTION_MODEL_PATH):
    print("   [GAGAL] File tidak ditemukan.")
else:
    try:
        # Hanya memuat bobot (state_dict), tidak perlu inisialisasi model penuh
        checkpoint = torch.load(EMOTION_MODEL_PATH, map_location=torch.device('cpu'))
        print("   [BERHASIL] File model emosi berhasil dimuat.")
        # Cek tipe data, harusnya dictionary
        if isinstance(checkpoint, dict):
            print("      - Tipe data: Dictionary (state_dict), ini benar.")
        else:
            print(f"      - Tipe data: {type(checkpoint)}, ini mungkin bukan state_dict.")
    except Exception as e:
        print(f"   [GAGAL] Terjadi error saat memuat model emosi: {e}")
        print("   --- TRACEBACK ERROR EMOSI ---")
        import traceback
        traceback.print_exc()
        print("   ---------------------------")


print("\n--- Pengecekan Selesai ---")
