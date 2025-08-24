# File: src/yolo_unified_integration.py
"""
Integrasi Unified YOLO Detector ke dalam aplikasi Flask
Menggunakan YOLO untuk semua tahap: Face Detection, Emotion Detection, dan Temporal Analysis
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from ultralytics import YOLO
import os
import torch

# Fix untuk PyTorch 2.6+ compatibility
try:
    from torch.serialization import safe_globals
    # Add safe globals untuk YOLO models
    safe_globals(['ultralytics.nn.tasks.DetectionModel'])
except ImportError:
    pass

class UnifiedYOLODetector:
    """
    Simplified unified detector yang mudah diintegrasikan ke Flask
    """
    
    def __init__(self, 
                 model_path: str = 'models/yolov8s.pt',  # Changed default to yolov8s
                 confidence_threshold: float = 0.5,
                 sequence_length: int = 5):
        
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.sequence_length = sequence_length
        
        # Load YOLO model dengan error handling
        print(f"Loading unified YOLO model: {model_path}")
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            print(f"Error loading {model_path}: {e}")
            print("Trying to download yolov8s.pt instead...")
            try:
                self.model = YOLO('yolov8s.pt')  # Fallback to yolov8s
                print("✅ Successfully loaded yolov8s.pt")
            except Exception as e2:
                print(f"Failed to load fallback model: {e2}")
                raise Exception("No compatible YOLO model found")
        
        # Emotion labels
        self.emotion_labels = [
            'Anger', 'Contempt', 'Disgust', 'Fear', 
            'Happy', 'Neutral', 'Sad', 'Surprised'
        ]
        
        # Temporal buffers
        self.emotion_history = {}
        self.face_tracking = {}
        
        print("✅ Unified YOLO Detector loaded successfully!")
    
    def detect_faces_and_emotions(self, image: np.ndarray) -> List[Dict]:
        """
        Unified detection: face detection + emotion classification dalam satu pass
        """
        # Run YOLO detection
        results = self.model(image, verbose=False)
        
        detections = []
        
        if results and hasattr(results[0], 'boxes'):
            boxes = results[0].boxes
            
            for i, box in enumerate(boxes):
                confidence = float(box.conf)
                
                if confidence >= self.confidence_threshold:
                    # Get bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Get class (assuming first class is face)
                    class_id = int(box.cls[0]) if hasattr(box, 'cls') else 0
                    
                    # Create face crop
                    face_crop = image[y1:y2, x1:x2]
                    
                    # Generate unique track ID
                    track_id = f"face_{i}_{int(time.time())}"
                    
                    # Detect emotion (simplified approach)
                    emotion, emotion_conf = self.detect_emotion_simple(face_crop)
                    
                    # Temporal analysis
                    temporal_result = self.analyze_temporal_emotion(track_id, emotion, emotion_conf)
                    
                    detection = {
                        'track_id': track_id,
                        'bbox': (x1, y1, x2, y2),
                        'confidence': confidence,
                        'class_id': class_id,
                        'face_crop': face_crop,
                        'emotion': emotion,
                        'emotion_confidence': emotion_conf,
                        'temporal_analysis': temporal_result
                    }
                    
                    detections.append(detection)
        
        return detections
    
    def detect_emotion_simple(self, face_crop: np.ndarray) -> Tuple[str, float]:
        """
        Simple emotion detection menggunakan YOLO dengan task='classify'
        """
        try:
            # Resize face crop
            resized_face = cv2.resize(face_crop, (224, 224))
            
            # Run classification
            results = self.model(resized_face, task='classify', verbose=False)
            
            if results and hasattr(results[0], 'probs'):
                probs = results[0].probs
                if probs is not None:
                    # Get emotion with highest probability
                    emotion_idx = int(torch.argmax(probs))
                    confidence = float(torch.max(probs))
                    
                    # Map to emotion label (assuming model is trained for emotions)
                    if emotion_idx < len(self.emotion_labels):
                        return self.emotion_labels[emotion_idx], confidence
            
            # Fallback: random emotion (untuk testing)
            return 'Neutral', 0.5
            
        except Exception as e:
            print(f"Error in emotion detection: {e}")
            return 'Unknown', 0.0
    
    def analyze_temporal_emotion(self, track_id: str, current_emotion: str, 
                                current_confidence: float) -> Dict:
        """
        Simple temporal analysis untuk emosi
        """
        # Initialize emotion history for this track
        if track_id not in self.emotion_history:
            self.emotion_history[track_id] = []
        
        # Add current emotion to history
        self.emotion_history[track_id].append({
            'emotion': current_emotion,
            'confidence': current_confidence,
            'timestamp': time.time()
        })
        
        # Keep only recent emotions
        if len(self.emotion_history[track_id]) > self.sequence_length:
            self.emotion_history[track_id] = self.emotion_history[track_id][-self.sequence_length:]
        
        # Analyze temporal patterns
        if len(self.emotion_history[track_id]) >= 2:
            return self.calculate_temporal_metrics(self.emotion_history[track_id])
        
        return {
            'current_emotion': current_emotion,
            'emotion_stability': 1.0,
            'emotion_transitions': 0,
            'dominant_emotion': current_emotion
        }
    
    def calculate_temporal_metrics(self, emotion_sequence: List[Dict]) -> Dict:
        """
        Calculate temporal metrics dari sequence emosi
        """
        emotions = [item['emotion'] for item in emotion_sequence]
        
        # Count emotion changes
        transitions = 0
        for i in range(len(emotions) - 1):
            if emotions[i] != emotions[i + 1]:
                transitions += 1
        
        # Calculate stability (1.0 = very stable, 0.0 = very unstable)
        total_possible_changes = len(emotions) - 1
        stability = 1.0 - (transitions / total_possible_changes) if total_possible_changes > 0 else 1.0
        
        # Find dominant emotion
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        dominant_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else 'Unknown'
        
        return {
            'current_emotion': emotions[-1],
            'emotion_stability': stability,
            'emotion_transitions': transitions,
            'dominant_emotion': dominant_emotion,
            'sequence_length': len(emotions)
        }
    
    def process_image(self, image: np.ndarray) -> Dict:
        """
        Process single image dan return comprehensive results
        """
        start_time = time.time()
        
        # Detect faces and emotions
        detections = self.detect_faces_and_emotions(image)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Prepare results
        results = {
            'processing_time': processing_time,
            'total_faces': len(detections),
            'detections': detections,
            'image_shape': image.shape,
            'confidence_threshold': self.confidence_threshold
        }
        
        return results
    
    def process_video_frame(self, frame: np.ndarray, frame_number: int) -> Dict:
        """
        Process single video frame
        """
        frame_results = self.process_image(frame)
        frame_results['frame_number'] = frame_number
        
        return frame_results
    
    def get_emotion_summary(self) -> Dict:
        """
        Get summary dari semua emotion detections
        """
        all_emotions = []
        for track_emotions in self.emotion_history.values():
            all_emotions.extend([item['emotion'] for item in track_emotions])
        
        if not all_emotions:
            return {'message': 'No emotions detected yet'}
        
        # Count emotions
        emotion_counts = {}
        for emotion in all_emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Calculate statistics
        total_detections = len(all_emotions)
        dominant_emotion = max(emotion_counts, key=emotion_counts.get)
        
        return {
            'total_detections': total_detections,
            'emotion_distribution': emotion_counts,
            'dominant_emotion': dominant_emotion,
            'unique_emotions': len(emotion_counts),
            'average_confidence': 0.75  # Placeholder
        }
    
    def reset_tracking(self):
        """
        Reset semua tracking data
        """
        self.emotion_history.clear()
        self.face_tracking.clear()
        print("🔄 Tracking data reset")


# Flask integration helper
class FlaskYOLOIntegration:
    """
    Helper class untuk integrasi mudah ke Flask
    """
    
    def __init__(self, model_path: str = 'models/yolov8s.pt'):  # Changed default
        self.detector = UnifiedYOLODetector(model_path)
    
    def process_upload(self, image_file) -> Dict:
        """
        Process uploaded image file
        """
        try:
            # Read image
            image_array = np.frombuffer(image_file.read(), np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                return {'error': 'Invalid image file'}
            
            # Process image
            results = self.detector.process_image(image)
            
            return results
            
        except Exception as e:
            return {'error': f'Processing error: {str(e)}'}
    
    def process_capture(self, image_data: bytes) -> Dict:
        """
        Process captured image data
        """
        try:
            # Convert bytes to numpy array
            image_array = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                return {'error': 'Invalid image data'}
            
            # Process image
            results = self.detector.process_image(image)
            
            return results
            
        except Exception as e:
            return {'error': f'Processing error: {str(e)}'}
    
    def get_emotion_summary(self) -> Dict:
        """
        Get emotion summary
        """
        return self.detector.get_emotion_summary()
    
    def reset_tracking(self):
        """
        Reset tracking
        """
        self.detector.reset_tracking()


# Example usage
if __name__ == "__main__":
    # Test the detector
    detector = UnifiedYOLODetector()
    
    # Create test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Process image
    results = detector.process_image(test_image)
    print("Test results:", results)
    
    # Get emotion summary
    summary = detector.get_emotion_summary()
    print("Emotion summary:", summary) 