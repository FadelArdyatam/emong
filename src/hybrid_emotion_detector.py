"""
Hybrid Emotion Detector
Mengkombinasikan rule-based approach dengan trained model untuk akurasi maksimal
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
import os
import json
from datetime import datetime

# Import existing detectors
from simple_emotion_detector import SimpleEmotionDetector
from trained_emotion_classifier import TrainedEmotionClassifier

class HybridEmotionDetector:
    """
    Hybrid emotion detector yang mengkombinasikan:
    1. Rule-based approach (OpenCV + MediaPipe)
    2. Trained model (FER-2013 CNN)
    3. Ensemble voting untuk hasil final
    """
    
    def __init__(self, 
                 trained_model_path: str = None,
                 use_mediapipe: bool = True,
                 ensemble_weight: float = 0.7):
        
        self.use_mediapipe = use_mediapipe
        self.ensemble_weight = ensemble_weight
        
        # Initialize detectors
        print("🔧 Initializing Hybrid Emotion Detector...")
        
        # 1. Rule-based detector (existing)
        self.rule_detector = SimpleEmotionDetector(use_mediapipe=use_mediapipe)
        print("✅ Rule-based detector initialized")
        
        # 2. Trained model detector
        self.trained_detector = TrainedEmotionClassifier(
            model_path=trained_model_path,
            model_type='fer2013'
        )
        print("✅ Trained model detector initialized")
        
        # 3. Performance tracking
        self.hybrid_results = []
        self.agreement_stats = {
            'total_comparisons': 0,
            'agreements': 0,
            'disagreements': 0
        }
        
        # 4. Confidence thresholds
        self.confidence_threshold = 0.6
        self.agreement_threshold = 0.8
        
        print("🚀 Hybrid Emotion Detector ready!")
    
    def detect_emotion_hybrid(self, face_crop: np.ndarray, landmarks=None) -> Dict:
        """
        Hybrid emotion detection dengan ensemble approach
        """
        start_time = time.time()
        
        try:
            # 1. Rule-based detection
            rule_emotion, rule_confidence = self.rule_detector.detect_emotion_simple(face_crop)
            
            # 2. Trained model detection
            trained_emotion, trained_confidence, trained_probs = self.trained_detector.predict_emotion(face_crop)
            
            # 3. Compare results
            comparison = self.trained_detector.compare_with_rule_based(face_crop, (rule_emotion, rule_confidence))
            
            # 4. Ensemble decision
            final_emotion, final_confidence, ensemble_method = self._ensemble_decision(
                rule_emotion, rule_confidence,
                trained_emotion, trained_confidence,
                comparison
            )
            
            # 5. Update statistics
            self._update_stats(comparison)
            
            # 6. Calculate processing time
            processing_time = time.time() - start_time
            
            # 7. Create result
            result = {
                'final_emotion': final_emotion,
                'final_confidence': final_confidence,
                'ensemble_method': ensemble_method,
                'rule_based': {
                    'emotion': rule_emotion,
                    'confidence': rule_confidence
                },
                'trained_model': {
                    'emotion': trained_emotion,
                    'confidence': trained_confidence,
                    'probabilities': trained_probs.tolist() if len(trained_probs) > 0 else []
                },
                'comparison': comparison,
                'processing_time': processing_time,
                'hybrid_confidence': self._calculate_hybrid_confidence(
                    rule_confidence, trained_confidence, comparison
                )
            }
            
            # 8. Store result
            self.hybrid_results.append(result)
            if len(self.hybrid_results) > 1000:
                self.hybrid_results = self.hybrid_results[-1000:]
            
            return result
            
        except Exception as e:
            print(f"❌ Error in hybrid detection: {e}")
            # Fallback to rule-based
            fallback_emotion, fallback_confidence = self.rule_detector.detect_emotion_simple(face_crop)
            return {
                'final_emotion': fallback_emotion,
                'final_confidence': fallback_confidence,
                'ensemble_method': 'fallback',
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def _ensemble_decision(self, rule_emotion: str, rule_conf: float,
                          trained_emotion: str, trained_conf: float,
                          comparison: Dict) -> Tuple[str, float, str]:
        """
        Make ensemble decision berdasarkan kedua predictions
        """
        try:
            # Case 1: High agreement between methods
            if comparison.get('agreement', False):
                # Use weighted average confidence
                final_confidence = (rule_conf * (1 - self.ensemble_weight) + 
                                  trained_conf * self.ensemble_weight)
                return rule_emotion, final_confidence, 'agreement'
            
            # Case 2: High confidence difference
            confidence_diff = comparison.get('confidence_difference', 0)
            if confidence_diff > 0.3:
                # Use the more confident prediction
                if trained_conf > rule_conf:
                    return trained_emotion, trained_conf, 'trained_high_confidence'
                else:
                    return rule_emotion, rule_conf, 'rule_high_confidence'
            
            # Case 3: Medium confidence, use ensemble
            if trained_conf > 0.7 and rule_conf > 0.5:
                # Both are confident, use trained model with boost
                final_confidence = min(1.0, trained_conf * 1.1)
                return trained_emotion, final_confidence, 'ensemble_boost'
            
            # Case 4: Low confidence, use rule-based as fallback
            if trained_conf < 0.4:
                return rule_emotion, rule_conf, 'rule_fallback'
            
            # Case 5: Default ensemble
            final_confidence = (rule_conf * (1 - self.ensemble_weight) + 
                              trained_conf * self.ensemble_weight)
            
            # Choose emotion based on confidence
            if trained_conf > rule_conf:
                return trained_emotion, final_confidence, 'ensemble_trained'
            else:
                return rule_emotion, final_confidence, 'ensemble_rule'
                
        except Exception as e:
            print(f"❌ Error in ensemble decision: {e}")
            # Fallback to rule-based
            return rule_emotion, rule_conf, 'error_fallback'
    
    def _calculate_hybrid_confidence(self, rule_conf: float, trained_conf: float, 
                                   comparison: Dict) -> float:
        """
        Calculate hybrid confidence score
        """
        try:
            # Base confidence
            base_confidence = (rule_conf * (1 - self.ensemble_weight) + 
                             trained_conf * self.ensemble_weight)
            
            # Agreement bonus
            if comparison.get('agreement', False):
                base_confidence *= 1.2
            
            # High confidence bonus
            if trained_conf > 0.8:
                base_confidence *= 1.1
            
            return min(1.0, base_confidence)
            
        except Exception as e:
            print(f"❌ Error calculating hybrid confidence: {e}")
            return (rule_conf + trained_conf) / 2
    
    def _update_stats(self, comparison: Dict):
        """Update agreement statistics"""
        self.agreement_stats['total_comparisons'] += 1
        
        if comparison.get('agreement', False):
            self.agreement_stats['agreements'] += 1
        else:
            self.agreement_stats['disagreements'] += 1
    
    def process_image_hybrid(self, image: np.ndarray) -> Dict:
        """
        Process image dengan hybrid approach
        """
        start_time = time.time()
        
        try:
            # Detect faces using rule-based detector
            faces = self.rule_detector.detect_faces(image)
            
            # Get facial landmarks if MediaPipe is available
            landmarks_results = None
            if self.use_mediapipe and len(faces) > 0:
                try:
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    landmarks_results = self.rule_detector.face_mesh.process(rgb_image)
                except Exception as e:
                    print(f"MediaPipe processing error: {e}")
            
            detections = []
            for i, face in enumerate(faces):
                face_id = face['face_id']
                face_crop = face['face_crop']
                
                # Get landmarks for this face
                face_landmarks = None
                if landmarks_results and landmarks_results.multi_face_landmarks:
                    if i < len(landmarks_results.multi_face_landmarks):
                        face_landmarks = landmarks_results.multi_face_landmarks[i]
                
                # Hybrid emotion detection
                hybrid_result = self.detect_emotion_hybrid(face_crop, face_landmarks)
                
                # Extract geometric features
                geometric_features = {}
                if face_landmarks:
                    geometric_features = self.rule_detector.extract_geometric_features(face_crop, face_landmarks)
                
                # Face quality assessment
                quality_score = self.rule_detector.assess_face_quality(face_crop)
                
                # Create detection result
                detection = {
                    'track_id': f"face_{face_id}",
                    'bbox': face['bbox'],
                    'confidence': face['confidence'],
                    'emotion': hybrid_result['final_emotion'],
                    'emotion_confidence': hybrid_result['final_confidence'],
                    'detection_method': 'hybrid',
                    'ensemble_method': hybrid_result['ensemble_method'],
                    'face_quality': quality_score,
                    'geometric_features': geometric_features,
                    'hybrid_analysis': {
                        'rule_based_emotion': hybrid_result['rule_based']['emotion'],
                        'rule_based_confidence': hybrid_result['rule_based']['confidence'],
                        'trained_model_emotion': hybrid_result['trained_model']['emotion'],
                        'trained_model_confidence': hybrid_result['trained_model']['confidence'],
                        'agreement': hybrid_result['comparison'].get('agreement', False),
                        'hybrid_confidence': hybrid_result['hybrid_confidence']
                    },
                    'image_width': face.get('image_width', image.shape[1]),
                    'image_height': face.get('image_height', image.shape[0])
                }
                
                detections.append(detection)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Return results
            results = {
                'detections': detections,
                'total_faces': len(detections),
                'processing_time': processing_time,
                'image_shape': image.shape,
                'timestamp': time.time(),
                'detection_method': 'hybrid',
                'mediapipe_enabled': self.use_mediapipe,
                'trained_model_loaded': self.trained_detector.model_loaded,
                'ensemble_weight': self.ensemble_weight,
                'agreement_stats': self.agreement_stats
            }
            
            return results
            
        except Exception as e:
            print(f"❌ Error in hybrid image processing: {e}")
            # Fallback to rule-based
            return self.rule_detector.process_image(image)
    
    def get_hybrid_stats(self) -> Dict:
        """Get hybrid detection statistics"""
        try:
            total_detections = len(self.hybrid_results)
            if total_detections == 0:
                return {
                    'total_detections': 0,
                    'agreement_rate': 0.0,
                    'ensemble_methods': {},
                    'accuracy_comparison': {}
                }
            
            # Agreement rate
            agreement_rate = self.agreement_stats['agreements'] / self.agreement_stats['total_comparisons']
            
            # Ensemble method distribution
            ensemble_methods = {}
            for result in self.hybrid_results:
                method = result.get('ensemble_method', 'unknown')
                if method not in ensemble_methods:
                    ensemble_methods[method] = 0
                ensemble_methods[method] += 1
            
            # Normalize ensemble methods
            for method in ensemble_methods:
                ensemble_methods[method] = ensemble_methods[method] / total_detections
            
            # Accuracy comparison
            accuracy_comparison = {
                'hybrid_avg_confidence': np.mean([r['final_confidence'] for r in self.hybrid_results]),
                'rule_based_avg_confidence': np.mean([r['hybrid_analysis']['rule_based_confidence'] for r in self.hybrid_results]),
                'trained_model_avg_confidence': np.mean([r['hybrid_analysis']['trained_model_confidence'] for r in self.hybrid_results]),
                'hybrid_confidence_boost': np.mean([r['hybrid_confidence'] for r in self.hybrid_results])
            }
            
            return {
                'total_detections': total_detections,
                'agreement_rate': agreement_rate,
                'ensemble_methods': ensemble_methods,
                'accuracy_comparison': accuracy_comparison,
                'agreement_stats': self.agreement_stats
            }
            
        except Exception as e:
            print(f"❌ Error getting hybrid stats: {e}")
            return {
                'total_detections': 0,
                'agreement_rate': 0.0,
                'ensemble_methods': {},
                'accuracy_comparison': {}
            }
    
    def save_hybrid_results(self, filepath: str = "hybrid_results.json"):
        """Save hybrid detection results"""
        try:
            save_data = {
                'hybrid_stats': self.get_hybrid_stats(),
                'trained_model_info': self.trained_detector.get_model_info(),
                'rule_based_info': {
                    'mediapipe_enabled': self.use_mediapipe,
                    'cascade_loaded': self.rule_detector.cascade is not None
                },
                'timestamp': datetime.now().isoformat(),
                'total_results': len(self.hybrid_results)
            }
            
            with open(filepath, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            print(f"✅ Hybrid results saved to: {filepath}")
            
        except Exception as e:
            print(f"❌ Error saving hybrid results: {e}")
    
    def reset_stats(self):
        """Reset all statistics"""
        self.hybrid_results.clear()
        self.agreement_stats = {
            'total_comparisons': 0,
            'agreements': 0,
            'disagreements': 0
        }
        print("🔄 Hybrid detector statistics reset") 