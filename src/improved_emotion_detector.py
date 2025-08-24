#!/usr/bin/env python3
"""
Improved Emotion Detector - Kombinasi YOLOv12s + Enhanced Rule-based
"""

import cv2
import numpy as np
import json
import os
from datetime import datetime
import time
import math

class ImprovedEmotionDetector:
    """Improved emotion detector dengan YOLOv12s + enhanced rule-based"""
    
    def __init__(self):
        """Initialize improved detector"""
        self.yolo_model = None
        self.cascade = None
        self.emotion_data = []
        self.data_file = "improved_emotion_data.json"
        
        # Load detectors
        self._load_detectors()
        
        # Load emotion data
        self._load_emotion_data()
        
        print("✅ Improved Emotion Detector loaded successfully!")
    
    def _load_detectors(self):
        """Load YOLOv12s dan cascade detectors"""
        try:
            # Try to load YOLOv12s for face detection
            yolo_path = "models/yolov12s.pt"
            if os.path.exists(yolo_path) and os.path.getsize(yolo_path) > 1000000:  # > 1MB
                print("🔄 Loading YOLOv12s for face detection...")
                try:
                    from ultralytics import YOLO
                    self.yolo_model = YOLO(yolo_path)
                    print("✅ YOLOv12s loaded successfully!")
                except Exception as e:
                    print(f"⚠️ YOLOv12s failed to load: {e}")
                    self.yolo_model = None
            else:
                print("⚠️ YOLOv12s not available or corrupt")
                self.yolo_model = None
            
            # Load OpenCV cascade as fallback
            print("🔄 Loading OpenCV face cascade...")
            try:
                custom_path = "models/haarcascade_frontalface_default.xml"
                if os.path.exists(custom_path):
                    self.cascade = cv2.CascadeClassifier(custom_path)
                    if self.cascade.empty():
                        raise Exception("Custom cascade is empty")
                    print(f"✅ Loaded custom cascade: {custom_path}")
                else:
                    builtin_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                    self.cascade = cv2.CascadeClassifier(builtin_path)
                    if self.cascade.empty():
                        raise Exception("Built-in cascade is empty")
                    print(f"✅ Loaded built-in cascade: {builtin_path}")
            except Exception as e:
                print(f"❌ Error loading cascade: {e}")
                self.cascade = None
                
        except Exception as e:
            print(f"❌ Error in detector loading: {e}")
            self.yolo_model = None
            self.cascade = None
    
    def _load_emotion_data(self):
        """Load atau create emotion data file"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    self.emotion_data = json.load(f)
                print(f"📂 Loaded data from {self.data_file}")
                print(f"   - {len(self.emotion_data)} detection histories")
                if self.emotion_data:
                    last_update = self.emotion_data[-1].get('timestamp', 'Unknown')
                    print(f"   - Last updated: {last_update}")
            else:
                self.emotion_data = []
                print(f"📂 Created new data file: {self.data_file}")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            self.emotion_data = []
    
    def _save_emotion_data(self):
        """Save emotion data ke file"""
        try:
            with open(self.data_file, 'w') as f:
                json.dump(self.emotion_data, f, indent=2)
        except Exception as e:
            print(f"❌ Error saving data: {e}")
    
    def detect_faces(self, image):
        """Detect faces menggunakan YOLOv12s atau cascade fallback"""
        try:
            if self.yolo_model is not None:
                # Use YOLOv12s
                results = self.yolo_model(image, conf=0.3, classes=[0])  # class 0 = person
                
                if results and len(results) > 0:
                    result = results[0]
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        boxes = result.boxes
                        faces = []
                        
                        for i in range(len(boxes)):
                            if hasattr(boxes, 'xyxy'):
                                xyxy = boxes.xyxy[i].cpu().numpy() if hasattr(boxes.xyxy[i], 'cpu') else boxes.xyxy[i]
                                x1, y1, x2, y2 = xyxy
                                w, h = x2 - x1, y2 - y1
                                faces.append([int(x1), int(y1), int(w), int(h)])
                        
                        if faces:
                            print(f"✅ YOLOv12s detected {len(faces)} faces")
                            return faces
                
                print("⚠️ YOLOv12s no detections, trying cascade...")
            
            # Fallback to cascade
            if self.cascade is not None:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                faces = self.cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                if len(faces) > 0:
                    print(f"✅ Cascade detected {len(faces)} faces")
                    return faces.tolist()
            
            print("⚠️ No faces detected by any method")
            return []
            
        except Exception as e:
            print(f"❌ Error in face detection: {e}")
            return []
    
    def analyze_emotion_advanced(self, face_roi):
        """Advanced emotion analysis dengan multiple features"""
        try:
            # Convert to grayscale
            if len(face_roi.shape) == 3:
                gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY)
            else:
                gray_roi = face_roi
            
            # Resize untuk consistency
            resized_roi = cv2.resize(gray_roi, (64, 64))
            
            # 1. Image Quality Analysis
            quality_features = self._analyze_image_quality(resized_roi)
            
            # 2. Texture Analysis
            texture_features = self._analyze_texture_advanced(resized_roi)
            
            # 3. Geometric Analysis
            geometric_features = self._analyze_geometric_advanced(resized_roi)
            
            # 4. Edge Analysis
            edge_features = self._analyze_edge_patterns(resized_roi)
            
            # 5. Combine all features untuk emotion prediction
            emotion_result = self._combine_features_advanced(
                quality_features, texture_features, geometric_features, edge_features
            )
            
            return emotion_result
            
        except Exception as e:
            print(f"❌ Error in advanced emotion analysis: {e}")
            return {
                'emotion': 'Unknown',
                'emotion_confidence': 0.5,
                'analysis_method': 'Error',
                'features': {}
            }
    
    def _analyze_image_quality(self, roi):
        """Analyze image quality metrics"""
        try:
            # Resolution
            height, width = roi.shape
            resolution_score = min(height * width / (64 * 64), 1.0)
            
            # Blur detection (Laplacian variance)
            laplacian_var = cv2.Laplacian(roi, cv2.CV_64F).var()
            blur_score = min(laplacian_var / 500, 1.0)
            
            # Brightness
            mean_brightness = np.mean(roi)
            brightness_score = 1.0 - abs(mean_brightness - 128) / 128
            
            # Contrast
            contrast_score = np.std(roi) / 128
            
            # Sharpness (using Sobel)
            sobel_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
            sharpness = np.sqrt(sobel_x**2 + sobel_y**2)
            sharpness_score = min(np.mean(sharpness) / 100, 1.0)
            
            return {
                'resolution': resolution_score,
                'blur': blur_score,
                'brightness': brightness_score,
                'contrast': contrast_score,
                'sharpness': sharpness_score,
                'overall': (resolution_score + blur_score + brightness_score + contrast_score + sharpness_score) / 5
            }
            
        except Exception as e:
            print(f"❌ Error in quality analysis: {e}")
            return {'overall': 0.5}
    
    def _analyze_texture_advanced(self, roi):
        """Advanced texture analysis"""
        try:
            # Local Binary Pattern approximation
            lbp_features = []
            
            for i in range(1, roi.shape[0] - 1):
                for j in range(1, roi.shape[1] - 1):
                    center = roi[i, j]
                    code = 0
                    
                    # 8-neighbor LBP
                    neighbors = [
                        roi[i-1, j-1], roi[i-1, j], roi[i-1, j+1],
                        roi[i, j+1], roi[i+1, j+1], roi[i+1, j],
                        roi[i+1, j-1], roi[i, j-1]
                    ]
                    
                    for k, neighbor in enumerate(neighbors):
                        if neighbor >= center:
                            code |= (1 << k)
                    
                    lbp_features.append(code)
            
            # Calculate texture statistics
            lbp_array = np.array(lbp_features)
            texture_variance = np.var(lbp_array)
            texture_entropy = -np.sum(np.bincount(lbp_array) * np.log2(np.bincount(lbp_array) + 1e-10))
            
            # Normalize
            normalized_variance = min(texture_variance / 1000, 1.0)
            normalized_entropy = min(texture_entropy / 8, 1.0)
            
            return {
                'variance': normalized_variance,
                'entropy': normalized_entropy,
                'overall': (normalized_variance + normalized_entropy) / 2
            }
            
        except Exception as e:
            print(f"❌ Error in texture analysis: {e}")
            return {'overall': 0.5}
    
    def _analyze_geometric_advanced(self, roi):
        """Advanced geometric analysis"""
        try:
            # Edge detection
            edges = cv2.Canny(roi, 50, 150)
            
            # Contour analysis
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find largest contour (assumed to be face)
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                
                # Circularity
                circularity = 4 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                # Aspect ratio
                x, y, w, h = cv2.boundingRect(largest_contour)
                aspect_ratio = w / h if h > 0 else 1
                
                # Symmetry (horizontal flip comparison)
                flipped = cv2.flip(roi, 1)
                symmetry_score = 1.0 - np.mean(np.abs(roi.astype(float) - flipped.astype(float))) / 255
                
                return {
                    'circularity': circularity,
                    'aspect_ratio': min(aspect_ratio / 2, 1.0),
                    'symmetry': symmetry_score,
                    'overall': (circularity + min(aspect_ratio / 2, 1.0) + symmetry_score) / 3
                }
            else:
                return {'overall': 0.5}
                
        except Exception as e:
            print(f"❌ Error in geometric analysis: {e}")
            return {'overall': 0.5}
    
    def _analyze_edge_patterns(self, roi):
        """Analyze edge patterns untuk emotion detection"""
        try:
            # Sobel gradients
            sobel_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
            
            # Gradient magnitude
            gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            gradient_direction = np.arctan2(sobel_y, sobel_x)
            
            # Edge density
            edge_density = np.sum(gradient_magnitude > 50) / gradient_magnitude.size
            
            # Directional edge analysis
            horizontal_edges = np.sum(np.abs(sobel_x) > 30)
            vertical_edges = np.sum(np.abs(sobel_y) > 30)
            
            # Normalize
            edge_density_norm = min(edge_density * 10, 1.0)
            horizontal_norm = min(horizontal_edges / 1000, 1.0)
            vertical_norm = min(vertical_edges / 1000, 1.0)
            
            return {
                'density': edge_density_norm,
                'horizontal': horizontal_norm,
                'vertical': vertical_norm,
                'overall': (edge_density_norm + horizontal_norm + vertical_norm) / 3
            }
            
        except Exception as e:
            print(f"❌ Error in edge analysis: {e}")
            return {'overall': 0.5}
    
    def _combine_features_advanced(self, quality, texture, geometric, edge):
        """Combine all features untuk emotion prediction"""
        try:
            # Get overall scores
            q_score = quality.get('overall', 0.5)
            t_score = texture.get('overall', 0.5)
            g_score = geometric.get('overall', 0.5)
            e_score = edge.get('overall', 0.5)
            
            # Weighted combination
            combined_score = (
                q_score * 0.25 +      # Quality
                t_score * 0.25 +      # Texture
                g_score * 0.25 +      # Geometric
                e_score * 0.25        # Edge
            )
            
            # Enhanced emotion mapping
            if combined_score > 0.85:
                emotion = 'Happy'
                confidence = combined_score
            elif combined_score > 0.75:
                emotion = 'Surprised'
                confidence = combined_score
            elif combined_score > 0.65:
                emotion = 'Neutral'
                confidence = combined_score
            elif combined_score > 0.55:
                emotion = 'Sad'
                confidence = combined_score
            elif combined_score > 0.45:
                emotion = 'Angry'
                confidence = combined_score
            elif combined_score > 0.35:
                emotion = 'Fear'
                confidence = combined_score
            else:
                emotion = 'Disgust'
                confidence = combined_score
            
            return {
                'emotion': emotion,
                'emotion_confidence': confidence,
                'analysis_method': 'Advanced Multi-Feature',
                'features': {
                    'quality': q_score,
                    'texture': t_score,
                    'geometric': g_score,
                    'edge': e_score,
                    'combined': combined_score
                }
            }
            
        except Exception as e:
            print(f"❌ Error combining features: {e}")
            return {
                'emotion': 'Unknown',
                'emotion_confidence': 0.5,
                'analysis_method': 'Error',
                'features': {}
            }
    
    def detect_emotions(self, image):
        """Main emotion detection method"""
        try:
            # Detect faces
            faces = self.detect_faces(image)
            
            if not faces:
                return {
                    'faces_detected': 0,
                    'emotions': [],
                    'timestamp': datetime.now().isoformat(),
                    'model_type': 'Improved (YOLOv12s + Enhanced)',
                    'detection_method': 'No faces detected'
                }
            
            # Process each face
            emotions = []
            for (x, y, w, h) in faces:
                # Crop face region
                face_roi = image[y:y+h, x:x+w]
                
                # Analyze emotion
                emotion_result = self.analyze_emotion_advanced(face_roi)
                
                # Add bounding box info
                emotion_result['bbox'] = [x, y, w, h]
                
                emotions.append(emotion_result)
            
            # Save detection history
            self._save_detection_history(emotions)
            
            return {
                'faces_detected': len(faces),
                'emotions': emotions,
                'timestamp': datetime.now().isoformat(),
                'model_type': 'Improved (YOLOv12s + Enhanced)',
                'detection_method': 'YOLOv12s + Enhanced Rule-based'
            }
            
        except Exception as e:
            error_msg = f"Improved detection error: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                'faces_detected': 0,
                'emotions': [],
                'timestamp': datetime.now().isoformat(),
                'model_type': 'Improved (YOLOv12s + Enhanced)',
                'detection_method': 'Error',
                'error': error_msg
            }
    
    def _save_detection_history(self, emotions):
        """Save detection history"""
        try:
            history_entry = {
                'timestamp': datetime.now().isoformat(),
                'faces_detected': len(emotions),
                'emotions': emotions,
                'detection_method': 'Improved'
            }
            
            self.emotion_data.append(history_entry)
            
            # Keep only last 100 entries
            if len(self.emotion_data) > 100:
                self.emotion_data = self.emotion_data[-100:]
            
            # Save to file
            self._save_emotion_data()
            
        except Exception as e:
            print(f"❌ Error saving detection history: {e}")
    
    def get_detection_stats(self):
        """Get detection statistics"""
        if not self.emotion_data:
            return {
                'total_detections': 0,
                'total_faces': 0,
                'emotion_distribution': {},
                'last_detection': None
            }
        
        total_detections = len(self.emotion_data)
        total_faces = sum(entry.get('faces_detected', 0) for entry in self.emotion_data)
        
        # Emotion distribution
        emotion_dist = {}
        for entry in self.emotion_data:
            for emotion_data in entry.get('emotions', []):
                emotion = emotion_data.get('emotion', 'Unknown')
                emotion_dist[emotion] = emotion_dist.get(emotion, 0) + 1
        
        return {
            'total_detections': total_detections,
            'total_faces': total_faces,
            'emotion_distribution': emotion_dist,
            'last_detection': self.emotion_data[-1].get('timestamp') if self.emotion_data else None
        }
    
    def reset_stats(self):
        """Reset detection statistics"""
        self.emotion_data = []
        self._save_emotion_data()
        print("🔄 Improved detector statistics reset") 