#!/usr/bin/env python3
"""
YOLO Hybrid Emotion Detector
Menggabungkan YOLO emotion detection dengan rule-based fallback
"""

import cv2
import numpy as np
import json
import os
from datetime import datetime
import time

from .yolo_emotion_detector import YOLOEmotionDetector
from .simple_emotion_detector import SimpleEmotionDetector

class YOLOHybridDetector:
    """Hybrid detector yang menggabungkan YOLO dan rule-based detection"""
    
    def __init__(self, yolo_model_path="models/yolo_emotion_detection_v8s.pt", 
                 yolo_weight=0.8, rule_weight=0.2, confidence_threshold=0.3):
        """
        Initialize hybrid detector
        
        Args:
            yolo_model_path: Path ke YOLO model
            yolo_weight: Weight untuk YOLO detection (0.0 - 1.0)
            rule_weight: Weight untuk rule-based detection (0.0 - 1.0)
            confidence_threshold: Minimum confidence untuk deteksi
        """
        self.yolo_weight = yolo_weight
        self.rule_weight = rule_weight
        self.confidence_threshold = confidence_threshold
        
        # Initialize detectors
        self.yolo_detector = None
        self.rule_detector = None
        
        # Detection history untuk temporal analysis
        self.detection_history = []
        self.data_file = "hybrid_emotion_data.json"
        
        # Load detectors
        self._load_detectors(yolo_model_path)
        
        # Load existing data
        self._load_emotion_data()
        
        print("🚀 YOLO Hybrid Detector initialized!")
        print(f"   - YOLO Weight: {self.yolo_weight}")
        print(f"   - Rule-based Weight: {self.rule_weight}")
        print(f"   - Confidence Threshold: {self.confidence_threshold}")
    
    def _load_detectors(self, yolo_model_path):
        """Load YOLO dan rule-based detectors"""
        try:
            # Load YOLO detector
            print("🔄 Loading YOLO detector...")
            self.yolo_detector = YOLOEmotionDetector(yolo_model_path)
            if self.yolo_detector.model is None:
                print("⚠️ YOLO detector failed to load, will use rule-based only")
                self.yolo_detector = None
            else:
                print("✅ YOLO detector loaded successfully")
        except Exception as e:
            print(f"❌ Error loading YOLO detector: {e}")
            self.yolo_detector = None
        
        try:
            # Load rule-based detector
            print("🔄 Loading rule-based detector...")
            self.rule_detector = SimpleEmotionDetector()
            print("✅ Rule-based detector loaded successfully")
        except Exception as e:
            print(f"❌ Error loading rule-based detector: {e}")
            self.rule_detector = None
        
        # Validate weights
        if self.yolo_detector is None:
            self.yolo_weight = 0.0
            self.rule_weight = 1.0
            print("⚠️ Adjusted weights: YOLO=0.0, Rule-based=1.0")
        elif self.rule_detector is None:
            self.yolo_weight = 1.0
            self.rule_weight = 0.0
            print("⚠️ Adjusted weights: YOLO=1.0, Rule-based=0.0")
    
    def _load_emotion_data(self):
        """Load atau create emotion data file"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    self.detection_history = json.load(f)
                print(f"📂 Loaded hybrid data from {self.data_file}")
                print(f"   - {len(self.detection_history)} detection histories")
                if self.detection_history:
                    last_update = self.detection_history[-1].get('timestamp', 'Unknown')
                    print(f"   - Last updated: {last_update}")
            else:
                self.detection_history = []
                print(f"📂 Created new hybrid data file: {self.data_file}")
        except Exception as e:
            print(f"❌ Error loading hybrid data: {e}")
            self.detection_history = []
    
    def _save_emotion_data(self):
        """Save emotion data ke file"""
        try:
            with open(self.data_file, 'w') as f:
                json.dump(self.detection_history, f, indent=2)
        except Exception as e:
            print(f"❌ Error saving hybrid data: {e}")
    
    def detect_emotions(self, image):
        """
        Detect emotions menggunakan hybrid approach
        
        Args:
            image: numpy array image (RGB)
        
        Returns:
            dict: Hasil deteksi hybrid
        """
        start_time = time.time()
        
        try:
            # Get YOLO detections
            yolo_results = None
            if self.yolo_detector and self.yolo_detector.model is not None:
                try:
                    yolo_results = self.yolo_detector.detect_emotions(image)
                except Exception as e:
                    print(f"⚠️ YOLO detection failed: {e}")
                    yolo_results = None
            
            # Get rule-based detections
            rule_results = None
            if self.rule_detector:
                try:
                    rule_results = self.rule_detector.detect_emotions(image)
                except Exception as e:
                    print(f"⚠️ Rule-based detection failed: {e}")
                    rule_results = None
            
            # Combine results using ensemble decision
            hybrid_results = self._ensemble_decision(yolo_results, rule_results, image)
            
            # Add processing time
            processing_time = time.time() - start_time
            hybrid_results['processing_time'] = processing_time
            hybrid_results['detection_method'] = 'Hybrid (YOLO + Rule-based)'
            
            # Save detection history
            self._save_detection_history(hybrid_results)
            
            return hybrid_results
            
        except Exception as e:
            error_msg = f"Hybrid detection error: {str(e)}"
            print(f"❌ {error_msg}")
            return self._get_error_result(error_msg)
    
    def _ensemble_decision(self, yolo_results, rule_results, image):
        """
        Combine YOLO dan rule-based results menggunakan ensemble decision
        
        Args:
            yolo_results: YOLO detection results
            rule_results: Rule-based detection results
            image: Original image
        
        Returns:
            dict: Combined hybrid results
        """
        try:
            # Initialize combined results
            combined_results = {
                'faces_detected': 0,
                'emotions': [],
                'timestamp': datetime.now().isoformat(),
                'model_type': 'Hybrid',
                'yolo_weight': self.yolo_weight,
                'rule_weight': self.rule_weight,
                'yolo_results': yolo_results,
                'rule_results': rule_results
            }
            
            # If no detections from either method
            if (yolo_results is None or yolo_results.get('faces_detected', 0) == 0) and \
               (rule_results is None or rule_results.get('faces_detected', 0) == 0):
                return combined_results
            
            # Process YOLO results
            yolo_emotions = []
            if yolo_results and yolo_results.get('faces_detected', 0) > 0:
                yolo_emotions = yolo_results.get('emotions', [])
                for emotion in yolo_emotions:
                    emotion['source'] = 'YOLO'
                    emotion['weight'] = self.yolo_weight
            
            # Process rule-based results
            rule_emotions = []
            if rule_results and rule_results.get('faces_detected', 0) > 0:
                rule_emotions = rule_results.get('emotions', [])
                for emotion in rule_emotions:
                    emotion['source'] = 'Rule-based'
                    emotion['weight'] = self.rule_weight
            
            # Combine emotions using weighted voting
            combined_emotions = self._combine_emotions(yolo_emotions, rule_emotions)
            
            # Update combined results
            combined_results['faces_detected'] = len(combined_emotions)
            combined_results['emotions'] = combined_emotions
            
            return combined_results
            
        except Exception as e:
            print(f"❌ Error in ensemble decision: {e}")
            return {
                'faces_detected': 0,
                'emotions': [],
                'timestamp': datetime.now().isoformat(),
                'model_type': 'Hybrid',
                'error': f'Ensemble decision error: {str(e)}'
            }
    
    def _combine_emotions(self, yolo_emotions, rule_emotions):
        """
        Combine emotions dari kedua detector menggunakan weighted voting
        
        Args:
            yolo_emotions: List of YOLO emotion detections
            rule_emotions: List of rule-based emotion detections
        
        Returns:
            list: Combined emotion detections
        """
        try:
            combined_emotions = []
            
            # If only one method has results, return those
            if not yolo_emotions and rule_emotions:
                return rule_emotions
            elif not rule_emotions and yolo_emotions:
                return yolo_emotions
            elif not yolo_emotions and not rule_emotions:
                return []
            
            # For each emotion detection, try to find matching face
            # This is a simplified approach - in practice you might want more sophisticated face matching
            
            # Add YOLO emotions
            for yolo_emotion in yolo_emotions:
                combined_emotions.append({
                    'bbox': yolo_emotion.get('bbox', []),
                    'emotion': yolo_emotion.get('emotion', 'Unknown'),
                    'emotion_confidence': yolo_emotion.get('emotion_confidence', 0.5),
                    'source': 'YOLO',
                    'weight': self.yolo_weight,
                    'combined_confidence': yolo_emotion.get('emotion_confidence', 0.5) * self.yolo_weight
                })
            
            # Add rule-based emotions (if no YOLO detections for that area)
            for rule_emotion in rule_emotions:
                # Check if this face area overlaps with any YOLO detection
                rule_bbox = rule_emotion.get('bbox', [])
                has_overlap = False
                
                for combined_emotion in combined_emotions:
                    if self._check_bbox_overlap(rule_bbox, combined_emotion.get('bbox', [])):
                        has_overlap = True
                        # Update confidence using weighted average
                        yolo_conf = combined_emotion.get('combined_confidence', 0)
                        rule_conf = rule_emotion.get('emotion_confidence', 0) * self.rule_weight
                        combined_emotion['combined_confidence'] = (yolo_conf + rule_conf) / 2
                        combined_emotion['source'] = 'Hybrid'
                        break
                
                if not has_overlap:
                    combined_emotions.append({
                        'bbox': rule_bbox,
                        'emotion': rule_emotion.get('emotion', 'Unknown'),
                        'emotion_confidence': rule_emotion.get('emotion_confidence', 0.5),
                        'source': 'Rule-based',
                        'weight': self.rule_weight,
                        'combined_confidence': rule_emotion.get('emotion_confidence', 0.5) * self.rule_weight
                    })
            
            # Filter by confidence threshold
            filtered_emotions = [
                emotion for emotion in combined_emotions 
                if emotion.get('combined_confidence', 0) >= self.confidence_threshold
            ]
            
            return filtered_emotions
            
        except Exception as e:
            print(f"❌ Error combining emotions: {e}")
            return []
    
    def _check_bbox_overlap(self, bbox1, bbox2, threshold=0.3):
        """
        Check if two bounding boxes overlap
        
        Args:
            bbox1: First bounding box [x, y, w, h]
            bbox2: Second bounding box [x, y, w, h]
            threshold: Minimum overlap ratio
        
        Returns:
            bool: True if overlap exceeds threshold
        """
        try:
            if len(bbox1) != 4 or len(bbox2) != 4:
                return False
            
            x1, y1, w1, h1 = bbox1
            x2, y2, w2, h2 = bbox2
            
            # Calculate intersection
            x_left = max(x1, x2)
            y_top = max(y1, y2)
            x_right = min(x1 + w1, x2 + w2)
            y_bottom = min(y1 + h1, y2 + h2)
            
            if x_right < x_left or y_bottom < y_top:
                return False
            
            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            bbox1_area = w1 * h1
            bbox2_area = w2 * h2
            
            # Calculate overlap ratio
            overlap_ratio = intersection_area / min(bbox1_area, bbox2_area)
            
            return overlap_ratio >= threshold
            
        except Exception as e:
            print(f"❌ Error checking bbox overlap: {e}")
            return False
    
    def _save_detection_history(self, detection_results):
        """Save detection history"""
        try:
            history_entry = {
                'timestamp': detection_results.get('timestamp', datetime.now().isoformat()),
                'faces_detected': detection_results.get('faces_detected', 0),
                'emotions': detection_results.get('emotions', []),
                'detection_method': detection_results.get('detection_method', 'Hybrid'),
                'processing_time': detection_results.get('processing_time', 0),
                'yolo_weight': self.yolo_weight,
                'rule_weight': self.rule_weight
            }
            
            self.detection_history.append(history_entry)
            
            # Keep only last 100 entries
            if len(self.detection_history) > 100:
                self.detection_history = self.detection_history[-100:]
            
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
            'model_type': 'Hybrid',
            'detection_method': 'Error',
            'error': error_msg,
            'processing_time': 0
        }
    
    def get_detection_stats(self):
        """Get detection statistics"""
        if not self.detection_history:
            return {
                'total_detections': 0,
                'total_faces': 0,
                'emotion_distribution': {},
                'last_detection': None,
                'yolo_usage': 0,
                'rule_usage': 0,
                'hybrid_usage': 0
            }
        
        total_detections = len(self.detection_history)
        total_faces = sum(entry.get('faces_detected', 0) for entry in self.detection_history)
        
        # Emotion distribution
        emotion_dist = {}
        yolo_usage = 0
        rule_usage = 0
        hybrid_usage = 0
        
        for entry in self.detection_history:
            for emotion_data in entry.get('emotions', []):
                emotion = emotion_data.get('emotion', 'Unknown')
                emotion_dist[emotion] = emotion_dist.get(emotion, 0) + 1
                
                # Count usage by source
                source = emotion_data.get('source', 'Unknown')
                if source == 'YOLO':
                    yolo_usage += 1
                elif source == 'Rule-based':
                    rule_usage += 1
                elif source == 'Hybrid':
                    hybrid_usage += 1
        
        return {
            'total_detections': total_detections,
            'total_faces': total_faces,
            'emotion_distribution': emotion_dist,
            'last_detection': self.detection_history[-1].get('timestamp') if self.detection_history else None,
            'yolo_usage': yolo_usage,
            'rule_usage': rule_usage,
            'hybrid_usage': hybrid_usage
        }
    
    def get_yolo_model_info(self):
        """Get YOLO model information"""
        if self.yolo_detector is None:
            return {
                'status': 'Not loaded',
                'model_path': 'N/A',
                'model_names': {},
                'error': 'YOLO detector failed to initialize'
            }
        
        try:
            return {
                'status': 'Loaded',
                'model_path': self.yolo_detector.model_path,
                'model_names': self.yolo_detector.model_names,
                'model_type': 'YOLOv8s',
                'task': 'Detection'
            }
        except Exception as e:
            return {
                'status': 'Error',
                'model_path': 'N/A',
                'model_names': {},
                'error': str(e)
            }
    
    def update_weights(self, yolo_weight, rule_weight):
        """Update detection weights"""
        if 0 <= yolo_weight <= 1 and 0 <= rule_weight <= 1:
            self.yolo_weight = yolo_weight
            self.rule_weight = rule_weight
            print(f"🔄 Updated weights: YOLO={yolo_weight}, Rule-based={rule_weight}")
            return True
        else:
            print("❌ Invalid weights: must be between 0 and 1")
            return False
    
    def update_confidence_threshold(self, threshold):
        """Update confidence threshold"""
        if 0 <= threshold <= 1:
            self.confidence_threshold = threshold
            print(f"🔄 Updated confidence threshold: {threshold}")
            return True
        else:
            print("❌ Invalid threshold: must be between 0 and 1")
            return False
    
    def reset_stats(self):
        """Reset detection statistics"""
        self.detection_history = []
        self._save_emotion_data()
        print("🔄 Hybrid detector statistics reset")
    
    def get_temporal_data(self, time_window=300):
        """
        Get temporal data untuk chart
        
        Args:
            time_window: Time window in seconds (default: 5 minutes)
        
        Returns:
            dict: Temporal data for charting
        """
        try:
            current_time = datetime.now()
            start_time = current_time.timestamp() - time_window
            
            # Filter recent detections
            recent_detections = []
            for entry in self.detection_history:
                try:
                    entry_time = datetime.fromisoformat(entry['timestamp']).timestamp()
                    if entry_time >= start_time:
                        recent_detections.append(entry)
                except:
                    continue
            
            # Prepare chart data
            chart_data = {
                'labels': [],
                'emotion_data': {},
                'face_count_data': [],
                'processing_time_data': []
            }
            
            # Group by time intervals
            for entry in recent_detections:
                timestamp = entry['timestamp']
                time_label = timestamp.split('T')[1][:8]  # HH:MM:SS
                
                if time_label not in chart_data['labels']:
                    chart_data['labels'].append(time_label)
                
                # Face count data
                chart_data['face_count_data'].append(entry.get('faces_detected', 0))
                
                # Processing time data
                chart_data['processing_time_data'].append(entry.get('processing_time', 0))
                
                # Emotion data
                for emotion_data in entry.get('emotions', []):
                    emotion = emotion_data.get('emotion', 'Unknown')
                    if emotion not in chart_data['emotion_data']:
                        chart_data['emotion_data'][emotion] = []
                    chart_data['emotion_data'][emotion].append(1)
            
            return chart_data
            
        except Exception as e:
            print(f"❌ Error getting temporal data: {e}")
            return {
                'labels': [],
                'emotion_data': {},
                'face_count_data': [],
                'processing_time_data': []
            } 