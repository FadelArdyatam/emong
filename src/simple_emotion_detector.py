#!/usr/bin/env python3
"""
Simple Emotion Detector - Rule-based fallback tanpa MediaPipe
Menggunakan OpenCV Haar Cascade + geometric analysis untuk emotion detection
"""

import cv2
import numpy as np
import json
import os
from datetime import datetime
import math

class SimpleEmotionDetector:
    """Simple rule-based emotion detector sebagai fallback"""
    
    def __init__(self):
        """Initialize simple emotion detector"""
        self.cascade = None
        self.emotion_data = []
        self.data_file = "emotion_data.json"
        
        # Load face cascade
        self._load_cascade()
        
        # Load emotion data
        self._load_emotion_data()
        
        print("✅ Simple Emotion Detector loaded successfully!")
    
    def _load_cascade(self):
        """Load OpenCV face cascade"""
        try:
            print("Loading OpenCV face cascade...")
            
            # Coba load custom cascade path
            custom_path = "models/haarcascade_frontalface_default.xml"
            if os.path.exists(custom_path):
                self.cascade = cv2.CascadeClassifier(custom_path)
                if self.cascade.empty():
                    raise Exception("Custom cascade is empty")
                print(f"✅ Loaded custom cascade: {custom_path}")
            else:
                # Fallback ke built-in cascade
                builtin_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.cascade = cv2.CascadeClassifier(builtin_path)
                if self.cascade.empty():
                    raise Exception("Built-in cascade is empty")
                print(f"✅ Loaded built-in cascade: {builtin_path}")
                
        except Exception as e:
            print(f"❌ Error loading cascade: {e}")
            self.cascade = None
    
    def _load_emotion_data(self):
        """Load atau create emotion data file"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    self.emotion_data = json.load(f)
                print(f"📂 Loaded data from {self.data_file}")
                print(f"   - {len(self.emotion_data)} face histories")
                if self.emotion_data:
                    last_update = self.emotion_data[-1].get('timestamp', 'Unknown')
                    print(f"   - Last updated: {last_update}")
            else:
                self.emotion_data = []
                print(f"📂 Created new emotion data file: {self.data_file}")
        except Exception as e:
            print(f"❌ Error loading emotion data: {e}")
            self.emotion_data = []
    
    def _save_emotion_data(self):
        """Save emotion data ke file"""
        try:
            with open(self.data_file, 'w') as f:
                json.dump(self.emotion_data, f, indent=2)
        except Exception as e:
            print(f"❌ Error saving emotion data: {e}")
    
    def detect_emotions(self, image):
        """
        Detect emotions menggunakan rule-based approach
        
        Args:
            image: numpy array image (RGB)
        
        Returns:
            dict: Hasil deteksi
        """
        if self.cascade is None:
            return self._get_error_result("Face cascade tidak tersedia")
        
        try:
            # Convert RGB ke grayscale untuk face detection
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Detect faces
            faces = self.cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            if len(faces) == 0:
                return self._get_no_detection_result()
            
            # Process setiap face
            emotions = []
            for (x, y, w, h) in faces:
                # Crop face region
                face_roi = gray[y:y+h, x:x+w]
                
                # Analyze face untuk emotion
                emotion_result = self._analyze_face_emotion(face_roi)
                
                # Add bounding box info
                emotion_result['bbox'] = [x, y, w, h]
                
                emotions.append(emotion_result)
            
            # Save detection history
            self._save_detection_history(emotions)
            
            return {
                'faces_detected': len(faces),
                'emotions': emotions,
                'timestamp': datetime.now().isoformat(),
                'model_type': 'Rule-based',
                'detection_method': 'OpenCV + Geometric Analysis'
            }
            
        except Exception as e:
            error_msg = f"Rule-based detection error: {str(e)}"
            print(f"❌ {error_msg}")
            return self._get_error_result(error_msg)
    
    def _analyze_face_emotion(self, face_roi):
        """
        Analyze face ROI untuk emotion detection
        
        Args:
            face_roi: Grayscale face image
        
        Returns:
            dict: Emotion analysis result
        """
        try:
            # Basic image quality metrics
            quality_score = self._assess_image_quality(face_roi)
            
            # Simple geometric features (tanpa MediaPipe)
            geometric_score = self._analyze_geometric_features(face_roi)
            
            # Texture analysis
            texture_score = self._analyze_texture(face_roi)
            
            # Combine scores untuk emotion prediction
            emotion_result = self._combine_scores(quality_score, geometric_score, texture_score)
            
            return emotion_result
            
        except Exception as e:
            print(f"❌ Error analyzing face: {e}")
            return {
                'emotion': 'Unknown',
                'emotion_confidence': 0.5,
                'analysis_method': 'Error',
                'quality_score': 0.0,
                'geometric_score': 0.0,
                'texture_score': 0.0
            }
    
    def _assess_image_quality(self, face_roi):
        """Assess image quality untuk face analysis"""
        try:
            # Resolution check
            height, width = face_roi.shape
            resolution_score = min(height * width / (64 * 64), 1.0)  # Normalize to 64x64
            
            # Blur detection (Laplacian variance)
            laplacian_var = cv2.Laplacian(face_roi, cv2.CV_64F).var()
            blur_score = min(laplacian_var / 500, 1.0)  # Normalize blur score
            
            # Brightness check
            mean_brightness = np.mean(face_roi)
            brightness_score = 1.0 - abs(mean_brightness - 128) / 128
            
            # Contrast check
            contrast_score = np.std(face_roi) / 128
            
            # Combine quality scores
            quality_score = (resolution_score + blur_score + brightness_score + contrast_score) / 4
            
            return {
                'resolution': resolution_score,
                'blur': blur_score,
                'brightness': brightness_score,
                'contrast': contrast_score,
                'overall': quality_score
            }
            
        except Exception as e:
            print(f"❌ Error assessing image quality: {e}")
            return {'overall': 0.5}
    
    def _analyze_geometric_features(self, face_roi):
        """Analyze geometric features tanpa MediaPipe"""
        try:
            # Simple edge detection
            edges = cv2.Canny(face_roi, 50, 150)
            
            # Count edge pixels (proxy untuk facial features)
            edge_density = np.sum(edges > 0) / (face_roi.shape[0] * face_roi.shape[1])
            
            # Simple symmetry analysis (horizontal flip comparison)
            flipped = cv2.flip(face_roi, 1)
            symmetry_score = 1.0 - np.mean(np.abs(face_roi.astype(float) - flipped.astype(float))) / 255
            
            # Basic shape analysis
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                shape_score = 4 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            else:
                shape_score = 0
            
            return {
                'edge_density': edge_density,
                'symmetry': symmetry_score,
                'shape': shape_score,
                'overall': (edge_density + symmetry_score + shape_score) / 3
            }
            
        except Exception as e:
            print(f"❌ Error analyzing geometric features: {e}")
            return {'overall': 0.5}
    
    def _analyze_texture(self, face_roi):
        """Analyze texture features menggunakan LBP"""
        try:
            # Simple texture analysis (Local Binary Pattern approximation)
            # Resize untuk consistency
            resized = cv2.resize(face_roi, (32, 32))
            
            # Calculate simple texture metrics
            # Horizontal gradient
            h_gradient = np.diff(resized, axis=1)
            h_texture = np.std(h_gradient)
            
            # Vertical gradient
            v_gradient = np.diff(resized, axis=0)
            v_texture = np.std(v_gradient)
            
            # Diagonal gradients
            d1_gradient = np.diff(np.diag(resized))
            d1_texture = np.std(d1_gradient)
            
            d2_gradient = np.diff(np.diag(np.fliplr(resized)))
            d2_texture = np.std(d2_gradient)
            
            # Combine texture scores
            texture_score = (h_texture + v_texture + d1_texture + d2_texture) / 4
            
            # Normalize
            normalized_texture = min(texture_score / 50, 1.0)
            
            return {
                'horizontal': h_texture,
                'vertical': v_texture,
                'diagonal1': d1_texture,
                'diagonal2': d2_texture,
                'overall': normalized_texture
            }
            
        except Exception as e:
            print(f"❌ Error analyzing texture: {e}")
            return {'overall': 0.5}
    
    def _combine_scores(self, quality_score, geometric_score, texture_score):
        """
        Combine semua scores untuk emotion prediction
        
        Args:
            quality_score: Image quality metrics
            geometric_score: Geometric feature scores
            texture_score: Texture analysis scores
        
        Returns:
            dict: Final emotion prediction
        """
        try:
            # Get overall scores
            q_score = quality_score.get('overall', 0.5)
            g_score = geometric_score.get('overall', 0.5)
            t_score = texture_score.get('overall', 0.5)
            
            # Weighted combination
            combined_score = (q_score * 0.4 + g_score * 0.4 + t_score * 0.2)
            
            # Simple emotion mapping berdasarkan combined score
            if combined_score > 0.8:
                emotion = 'Happy'
                confidence = combined_score
            elif combined_score > 0.6:
                emotion = 'Neutral'
                confidence = combined_score
            elif combined_score > 0.4:
                emotion = 'Sad'
                confidence = combined_score
            else:
                emotion = 'Unknown'
                confidence = 0.5
            
            return {
                'emotion': emotion,
                'emotion_confidence': confidence,
                'analysis_method': 'Rule-based + Geometric',
                'quality_score': q_score,
                'geometric_score': g_score,
                'texture_score': t_score,
                'combined_score': combined_score
            }
            
        except Exception as e:
            print(f"❌ Error combining scores: {e}")
            return {
                'emotion': 'Unknown',
                'emotion_confidence': 0.5,
                'analysis_method': 'Error',
                'quality_score': 0.0,
                'geometric_score': 0.0,
                'texture_score': 0.0,
                'combined_score': 0.0
            }
    
    def _save_detection_history(self, emotions):
        """Save detection history"""
        try:
            history_entry = {
                'timestamp': datetime.now().isoformat(),
                'faces_detected': len(emotions),
                'emotions': emotions,
                'detection_method': 'Rule-based'
            }
            
            self.emotion_data.append(history_entry)
            
            # Keep only last 100 entries
            if len(self.emotion_data) > 100:
                self.emotion_data = self.emotion_data[-100:]
            
            # Save to file
            self._save_emotion_data()
            
        except Exception as e:
            print(f"❌ Error saving detection history: {e}")
    
    def _get_error_result(self, error_msg):
        """Generate error result"""
        return {
            'faces_detected': 0,
            'emotions': [],
            'timestamp': datetime.now().isoformat(),
            'model_type': 'Rule-based',
            'detection_method': 'Error',
            'error': error_msg
        }
    
    def _get_no_detection_result(self):
        """Generate no detection result"""
        return {
            'faces_detected': 0,
            'emotions': [],
            'timestamp': datetime.now().isoformat(),
            'model_type': 'Rule-based',
            'detection_method': 'No faces detected'
        }
    
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
        print("🔄 Rule-based detector statistics reset") 