#!/usr/bin/env python3
"""
YOLO Emotion Detector - Integrasi dengan sistem emotion detection
Menggunakan model yolo_emotion_detection_v8s.pt untuk deteksi wajah dan emotion
"""

import cv2
import numpy as np
from ultralytics import YOLO
import json
from datetime import datetime
import os
import torch

class YOLOEmotionDetector:
    """YOLO-based emotion detector untuk sistem hybrid"""
    
    def __init__(self, model_path="models/yolo_emotion_detection_v8s.pt"):
        """
        Initialize YOLO emotion detector
        
        Args:
            model_path: Path ke model YOLO (.pt file)
        """
        self.model_path = model_path
        self.model = None
        self.model_names = {}
        self.confidence_threshold = 0.3
        self.detection_history = []
        self.performance_stats = {
            'total_detections': 0,
            'successful_detections': 0,
            'failed_detections': 0,
            'avg_confidence': 0.0,
            'total_inference_time': 0.0
        }
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load YOLO model dengan PyTorch compatibility fix"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model tidak ditemukan: {self.model_path}")
            
            print(f"🔄 Loading YOLO model: {self.model_path}")
            
            # Fix untuk PyTorch 2.6 compatibility
            try:
                # Coba load dengan cara normal
                self.model = YOLO(self.model_path)
                self.model_names = self.model.names
            except Exception as e:
                if "WeightsUnpickler error" in str(e) or "Unsupported global" in str(e):
                    print("⚠️  PyTorch compatibility issue detected, trying alternative loading...")
                    
                    # Coba dengan safe globals
                    try:
                        from ultralytics.nn.tasks import DetectionModel
                        torch.serialization.add_safe_globals([DetectionModel])
                        self.model = YOLO(self.model_path)
                        self.model_names = self.model.names
                        print("✅ Model loaded with safe globals fix")
                    except Exception as e2:
                        print(f"❌ Safe globals fix failed: {e2}")
                        
                        # Fallback: coba dengan weights_only=False
                        try:
                            print("🔄 Trying weights_only=False approach...")
                            # Buat temporary YOLO instance dan load weights manually
                            self.model = YOLO('yolov8s.pt')  # Load default model first
                            # Load custom weights
                            self.model = YOLO(self.model_path, task='detect')
                            self.model_names = self.model.names
                            print("✅ Model loaded with weights_only=False approach")
                        except Exception as e3:
                            print(f"❌ All loading methods failed: {e3}")
                            raise e3
                else:
                    raise e
            
            print(f"✅ YOLO Model loaded successfully!")
            print(f"📊 Classes: {self.model_names}")
            print(f"🔢 Number of classes: {len(self.model_names)}")
            
        except Exception as e:
            print(f"❌ Error loading YOLO model: {e}")
            self.model = None
            self.model_names = {}
    
    def detect_emotions(self, image, confidence_threshold=None):
        """
        Detect emotions menggunakan YOLO
        
        Args:
            image: numpy array image (RGB)
            confidence_threshold: Threshold confidence (optional)
        
        Returns:
            dict: Hasil deteksi yang sudah dikonversi
        """
        if self.model is None:
            return self._get_error_result("Model YOLO tidak tersedia")
        
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
        
        start_time = datetime.now()
        
        try:
            # Inference
            results = self.model(image, conf=self.confidence_threshold)
            
            # Ambil result pertama
            if results and len(results) > 0:
                result = results[0]
                converted = self._convert_yolo_result(result)
                
                # Update performance stats
                inference_time = (datetime.now() - start_time).total_seconds()
                self._update_performance_stats(converted, inference_time)
                
                return converted
            else:
                return self._get_no_detection_result()
                
        except Exception as e:
            error_msg = f"YOLO inference error: {str(e)}"
            print(f"❌ {error_msg}")
            return self._get_error_result(error_msg)
    
    def _convert_yolo_result(self, yolo_result):
        """
        Konversi output YOLO ke format yang bisa digunakan sistem kita
        
        Args:
            yolo_result: Hasil dari model YOLO
        
        Returns:
            dict: Format yang sudah dikonversi
        """
        converted_result = {
            'faces_detected': 0,
            'emotions': [],
            'timestamp': datetime.now().isoformat(),
            'model_type': 'YOLO',
            'confidence_threshold': self.confidence_threshold,
            'raw_yolo_output': {}
        }
        
        if hasattr(yolo_result, 'boxes') and yolo_result.boxes is not None:
            boxes = yolo_result.boxes
            
            if len(boxes) > 0:
                converted_result['faces_detected'] = len(boxes)
                
                # Konversi setiap deteksi
                for i in range(len(boxes)):
                    detection = {}
                    
                    # Bounding box
                    if hasattr(boxes, 'xyxy'):
                        xyxy = boxes.xyxy[i].cpu().numpy() if hasattr(boxes.xyxy[i], 'cpu') else boxes.xyxy[i]
                        detection['bbox'] = {
                            'x1': float(xyxy[0]),
                            'y1': float(xyxy[1]),
                            'x2': float(xyxy[2]),
                            'y2': float(xyxy[3])
                        }
                    
                    # Confidence
                    if hasattr(boxes, 'conf'):
                        conf = boxes.conf[i].cpu().numpy() if hasattr(boxes.conf[i], 'cpu') else boxes.conf[i]
                        detection['confidence'] = float(conf)
                    
                    # Class/Emotion
                    if hasattr(boxes, 'cls'):
                        cls = boxes.cls[i].cpu().numpy() if hasattr(boxes.cls[i], 'cpu') else boxes.cls[i]
                        class_id = int(cls)
                        emotion_name = self.model_names.get(class_id, f"Unknown_{class_id}")
                        detection['emotion'] = emotion_name
                        detection['emotion_id'] = class_id
                    
                    # Raw data untuk debugging
                    if hasattr(boxes, 'data'):
                        raw_data = boxes.data[i].cpu().numpy() if hasattr(boxes.data[i], 'cpu') else boxes.data[i]
                        detection['raw_data'] = raw_data.tolist() if hasattr(raw_data, 'tolist') else raw_data
                    
                    converted_result['emotions'].append(detection)
                
                # Raw YOLO output untuk debugging
                converted_result['raw_yolo_output'] = {
                    'boxes_count': len(boxes),
                    'boxes_data_shape': str(boxes.data.shape) if hasattr(boxes.data, 'shape') else 'No shape',
                    'boxes_data_type': str(type(boxes.data))
                }
        
        return converted_result
    
    def _get_error_result(self, error_msg):
        """Generate error result"""
        return {
            'faces_detected': 0,
            'emotions': [],
            'timestamp': datetime.now().isoformat(),
            'model_type': 'YOLO',
            'confidence_threshold': self.confidence_threshold,
            'raw_yolo_output': {'error': error_msg}
        }
    
    def _get_no_detection_result(self):
        """Generate no detection result"""
        return {
            'faces_detected': 0,
            'emotions': [],
            'timestamp': datetime.now().isoformat(),
            'model_type': 'YOLO',
            'confidence_threshold': self.confidence_threshold,
            'raw_yolo_output': {'info': 'No detections found'}
        }
    
    def _update_performance_stats(self, result, inference_time):
        """Update performance statistics"""
        self.performance_stats['total_detections'] += 1
        self.performance_stats['total_inference_time'] += inference_time
        
        if result['faces_detected'] > 0:
            self.performance_stats['successful_detections'] += 1
            
            # Update average confidence
            total_conf = sum(emotion['confidence'] for emotion in result['emotions'])
            avg_conf = total_conf / len(result['emotions'])
            
            current_avg = self.performance_stats['avg_confidence']
            total_success = self.performance_stats['successful_detections']
            
            # Calculate running average
            self.performance_stats['avg_confidence'] = (current_avg * (total_success - 1) + avg_conf) / total_success
        else:
            self.performance_stats['failed_detections'] += 1
        
        # Store in history
        self.detection_history.append({
            'timestamp': result['timestamp'],
            'faces_detected': result['faces_detected'],
            'inference_time': inference_time,
            'success': result['faces_detected'] > 0
        })
        
        # Keep only last 100 entries
        if len(self.detection_history) > 100:
            self.detection_history = self.detection_history[-100:]
    
    def get_model_info(self):
        """Get info tentang model"""
        return {
            'model_type': 'YOLO',
            'model_path': self.model_path,
            'classes': self.model_names,
            'num_classes': len(self.model_names),
            'confidence_threshold': self.confidence_threshold,
            'model_loaded': self.model is not None
        }
    
    def update_confidence_threshold(self, new_threshold):
        """Update confidence threshold"""
        if 0.0 <= new_threshold <= 1.0:
            self.confidence_threshold = new_threshold
            return True
        return False
    
    def get_performance_stats(self):
        """Get performance statistics"""
        stats = self.performance_stats.copy()
        
        if stats['total_detections'] > 0:
            stats['success_rate'] = stats['successful_detections'] / stats['total_detections']
            stats['avg_inference_time'] = stats['total_inference_time'] / stats['total_detections']
        else:
            stats['success_rate'] = 0.0
            stats['avg_inference_time'] = 0.0
        
        return stats
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.performance_stats = {
            'total_detections': 0,
            'successful_detections': 0,
            'failed_detections': 0,
            'avg_confidence': 0.0,
            'total_inference_time': 0.0
        }
        self.detection_history = []
    
    def is_available(self):
        """Check if YOLO model is available"""
        return self.model is not None and len(self.model_names) > 0 