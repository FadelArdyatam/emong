# File: src/yolo_unified_temporal.py
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from collections import deque
import os
from ultralytics import YOLO
import time
from typing import List, Tuple, Dict, Optional

class YOLOTemporalEmotionDetector:
    """
    Unified YOLO-based system untuk:
    1. Face Detection (YOLO)
    2. Emotion Detection (YOLO + Custom Head)
    3. Temporal Analysis (YOLO + LSTM/Transformer)
    """
    
    def __init__(self, 
                 face_model_path: str = 'models/yolov12s.pt',
                 emotion_model_path: str = 'models/yolov12s.pt',
                 temporal_model_path: str = 'models/yolov12s.pt',
                 sequence_length: int = 10,
                 device: str = 'auto'):
        
        self.sequence_length = sequence_length
        self.device = torch.device('cuda' if torch.cuda.is_available() and device != 'cpu' else 'cpu')
        
        # Load YOLO models
        print(f"Loading YOLO models on {self.device}...")
        
        # 1. Face Detection Model
        self.face_detector = YOLO(face_model_path)
        self.face_detector.to(self.device)
        
        # 2. Emotion Detection Model (YOLO dengan custom emotion head)
        self.emotion_detector = YOLO(emotion_model_path)
        self.emotion_detector.to(self.device)
        
        # 3. Temporal Model (YOLO + LSTM)
        self.temporal_detector = YOLOTemporalModel(
            yolo_model_path=temporal_model_path,
            sequence_length=sequence_length,
            device=self.device
        )
        
        # Buffer untuk temporal analysis
        self.face_sequences = {}  # track_id -> deque of face crops
        self.emotion_history = {}  # track_id -> deque of emotions
        
        # Emotion labels
        self.emotion_labels = [
            'Anger', 'Contempt', 'Disgust', 'Fear', 
            'Happy', 'Neutral', 'Sad', 'Surprised'
        ]
        
        print("✅ YOLO Unified Temporal Emotion Detector loaded successfully!")
    
    def detect_faces(self, image: np.ndarray, confidence: float = 0.5) -> List[Dict]:
        """
        Deteksi wajah menggunakan YOLO
        """
        results = self.face_detector(image, verbose=False)
        faces = []
        
        if results and hasattr(results[0], 'boxes'):
            boxes = results[0].boxes
            for box in boxes:
                if float(box.conf) >= confidence:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    faces.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': float(box.conf),
                        'face_crop': image[y1:y2, x1:x2]
                    })
        
        return faces
    
    def detect_emotion_single(self, face_crop: np.ndarray) -> Tuple[str, float]:
        """
        Deteksi emosi untuk single face crop menggunakan YOLO + custom head
        """
        # Preprocess face crop
        processed_face = self.preprocess_face(face_crop)
        
        # Run emotion detection
        results = self.emotion_detector(processed_face, verbose=False)
        
        # Extract emotion prediction (assuming custom emotion head)
        if results and hasattr(results[0], 'probs'):
            probs = results[0].probs
            if probs is not None:
                emotion_idx = int(torch.argmax(probs))
                confidence = float(torch.max(probs))
                return self.emotion_labels[emotion_idx], confidence
        
        return 'Unknown', 0.0
    
    def analyze_temporal_emotion(self, track_id: str, current_emotion: str, 
                                current_confidence: float) -> Dict:
        """
        Analisis temporal emosi menggunakan YOLO + LSTM
        """
        # Update emotion history
        if track_id not in self.emotion_history:
            self.emotion_history[track_id] = deque(maxlen=self.sequence_length)
        
        self.emotion_history[track_id].append({
            'emotion': current_emotion,
            'confidence': current_confidence,
            'timestamp': time.time()
        })
        
        # Jika sequence sudah cukup panjang, lakukan temporal analysis
        if len(self.emotion_history[track_id]) >= self.sequence_length:
            return self.temporal_detector.analyze_sequence(
                list(self.emotion_history[track_id])
            )
        
        return {
            'current_emotion': current_emotion,
            'confidence': current_confidence,
            'temporal_analysis': 'Insufficient data',
            'emotion_stability': 0.0
        }
    
    def preprocess_face(self, face_crop: np.ndarray) -> np.ndarray:
        """
        Preprocess face crop untuk emotion detection
        """
        # Resize ke ukuran yang diharapkan model
        target_size = (224, 224)
        resized = cv2.resize(face_crop, target_size)
        
        # Normalisasi
        normalized = resized.astype(np.float32) / 255.0
        
        # Convert ke tensor format yang diharapkan YOLO
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
        
        return tensor.numpy()
    
    def process_frame(self, image: np.ndarray, confidence: float = 0.5) -> List[Dict]:
        """
        Process single frame: detect faces, emotions, dan temporal analysis
        """
        # 1. Face Detection
        faces = self.detect_faces(image, confidence)
        
        results = []
        for i, face in enumerate(faces):
            track_id = f"face_{i}_{int(time.time())}"
            
            # 2. Emotion Detection
            emotion, emotion_conf = self.detect_emotion_single(face['face_crop'])
            
            # 3. Temporal Analysis
            temporal_result = self.analyze_temporal_emotion(track_id, emotion, emotion_conf)
            
            # Combine results
            result = {
                'track_id': track_id,
                'bbox': face['bbox'],
                'face_confidence': face['confidence'],
                'emotion': emotion,
                'emotion_confidence': emotion_conf,
                'temporal_analysis': temporal_result
            }
            
            results.append(result)
        
        return results
    
    def process_video_sequence(self, frames: List[np.ndarray], 
                             confidence: float = 0.5) -> List[List[Dict]]:
        """
        Process sequence of frames untuk analisis temporal yang lebih baik
        """
        all_results = []
        
        for frame in frames:
            frame_results = self.process_frame(frame, confidence)
            all_results.append(frame_results)
        
        # Additional temporal analysis across frames
        temporal_insights = self.analyze_cross_frame_temporal(all_results)
        
        return all_results, temporal_insights
    
    def analyze_cross_frame_temporal(self, frame_results: List[List[Dict]]) -> Dict:
        """
        Analisis temporal across multiple frames
        """
        if not frame_results:
            return {}
        
        # Collect all emotions across frames
        all_emotions = []
        for frame in frame_results:
            for detection in frame:
                all_emotions.append(detection['emotion'])
        
        # Calculate emotion transitions
        emotion_transitions = {}
        for i in range(len(all_emotions) - 1):
            current = all_emotions[i]
            next_emotion = all_emotions[i + 1]
            transition = f"{current}->{next_emotion}"
            emotion_transitions[transition] = emotion_transitions.get(transition, 0) + 1
        
        # Calculate emotion stability
        emotion_counts = {}
        for emotion in all_emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        total_detections = len(all_emotions)
        emotion_stability = max(emotion_counts.values()) / total_detections if total_detections > 0 else 0
        
        return {
            'emotion_transitions': emotion_transitions,
            'emotion_stability': emotion_stability,
            'total_frames': len(frame_results),
            'total_detections': total_detections,
            'dominant_emotion': max(emotion_counts, key=emotion_counts.get) if emotion_counts else 'Unknown'
        }


class YOLOTemporalModel(nn.Module):
    """
    YOLO + LSTM untuk analisis temporal emosi
    """
    
    def __init__(self, yolo_model_path: str, sequence_length: int, device: str):
        super(YOLOTemporalModel, self).__init__()
        
        self.sequence_length = sequence_length
        self.device = device
        
        # Load YOLO base model
        self.yolo_base = YOLO(yolo_model_path)
        self.yolo_base.to(device)
        
        # LSTM untuk temporal modeling
        self.lstm = nn.LSTM(
            input_size=8,  # 8 emotions
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        
        # Emotion transition predictor
        self.emotion_predictor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 8)  # 8 emotions
        )
        
        self.to(device)
    
    def analyze_sequence(self, emotion_sequence: List[Dict]) -> Dict:
        """
        Analisis sequence emosi menggunakan LSTM
        """
        if len(emotion_sequence) < self.sequence_length:
            return {'error': 'Insufficient sequence length'}
        
        # Convert emotions to one-hot encoding
        emotion_onehot = []
        for item in emotion_sequence:
            emotion_idx = self.get_emotion_index(item['emotion'])
            onehot = [0] * 8
            onehot[emotion_idx] = 1
            emotion_onehot.append(onehot)
        
        # Convert to tensor
        emotion_tensor = torch.tensor(emotion_onehot, dtype=torch.float32).unsqueeze(0)
        emotion_tensor = emotion_tensor.to(self.device)
        
        # LSTM processing
        with torch.no_grad():
            lstm_out, _ = self.lstm(emotion_tensor)
            
            # Predict next emotion
            next_emotion_logits = self.emotion_predictor(lstm_out[:, -1, :])
            next_emotion_probs = torch.softmax(next_emotion_logits, dim=1)
            
            # Get prediction
            predicted_emotion_idx = torch.argmax(next_emotion_probs).item()
            predicted_confidence = torch.max(next_emotion_probs).item()
            
            # Calculate emotion stability
            emotion_stability = self.calculate_emotion_stability(emotion_sequence)
        
        return {
            'predicted_next_emotion': self.get_emotion_label(predicted_emotion_idx),
            'prediction_confidence': predicted_confidence,
            'emotion_stability': emotion_stability,
            'sequence_length': len(emotion_sequence)
        }
    
    def get_emotion_index(self, emotion: str) -> int:
        """Get emotion index from label"""
        emotion_labels = [
            'Anger', 'Contempt', 'Disgust', 'Fear', 
            'Happy', 'Neutral', 'Sad', 'Surprised'
        ]
        return emotion_labels.index(emotion) if emotion in emotion_labels else 0
    
    def get_emotion_label(self, index: int) -> str:
        """Get emotion label from index"""
        emotion_labels = [
            'Anger', 'Contempt', 'Disgust', 'Fear', 
            'Happy', 'Neutral', 'Sad', 'Surprised'
        ]
        return emotion_labels[index] if 0 <= index < len(emotion_labels) else 'Unknown'
    
    def calculate_emotion_stability(self, emotion_sequence: List[Dict]) -> float:
        """Calculate emotion stability score"""
        if len(emotion_sequence) < 2:
            return 0.0
        
        # Count emotion changes
        changes = 0
        for i in range(len(emotion_sequence) - 1):
            if emotion_sequence[i]['emotion'] != emotion_sequence[i + 1]['emotion']:
                changes += 1
        
        # Stability = 1 - (changes / total_possible_changes)
        total_possible_changes = len(emotion_sequence) - 1
        stability = 1.0 - (changes / total_possible_changes) if total_possible_changes > 0 else 1.0
        
        return stability


# Utility functions
def create_yolo_emotion_head():
    """
    Create custom emotion detection head untuk YOLO
    """
    # Ini bisa diimplementasikan sebagai custom YOLO head
    # atau menggunakan YOLO dengan task='classify'
    pass


def test_unified_detector():
    """
    Test function untuk unified detector
    """
    detector = YOLOTemporalEmotionDetector()
    
    # Test dengan dummy image
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    results = detector.process_frame(dummy_image)
    print(f"Detection results: {results}")


if __name__ == "__main__":
    test_unified_detector() 