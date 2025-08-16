# File: src/model_loader.py
from ultralytics import YOLO

def load_model(model_path):
    """
    Memuat model YOLO dari path yang diberikan.
    
    Args:
        model_path (str): Path ke file model .pt
    
    Returns:
        model: Model YOLO yang sudah dimuat
    """
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        raise Exception(f"Tidak dapat memuat model dari {model_path}: {str(e)}")
