"""
Trained Emotion Classifier menggunakan model yang sudah di-training
Integrasi dengan sistem existing untuk hybrid approach
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from typing import Dict, Tuple, Optional, List
import json
from datetime import datetime

class TrainedEmotionClassifier:
    """
    Emotion classifier menggunakan pre-trained model
    Support untuk FER-2013 dan custom models
    """
    
    def __init__(self, 
                 model_path: str = None,
                 model_type: str = 'fer2013',
                 input_size: Tuple[int, int] = (48, 48),
                 use_gpu: bool = True):
        
        self.model_path = model_path
        self.model_type = model_type
        self.input_size = input_size
        self.use_gpu = use_gpu
        
        # Emotion labels untuk FER-2013
        self.emotion_labels = {
            0: 'Angry',
            1: 'Disgust', 
            2: 'Fear',
            3: 'Happy',
            4: 'Sad',
            5: 'Surprise',
            6: 'Neutral'
        }
        
        # Initialize model
        self.model = None
        self.model_loaded = False
        self.load_model()
        
        # Performance tracking
        self.inference_times = []
        self.accuracy_history = []
        
    def load_model(self):
        """Load pre-trained emotion classification model"""
        try:
            if self.model_path and os.path.exists(self.model_path):
                print(f"🔄 Loading model from: {self.model_path}")
                self.model = keras.models.load_model(self.model_path)
                self.model_loaded = True
                print("✅ Model loaded successfully!")
                
                # Model summary
                print(f"📊 Model Architecture:")
                print(f"   - Input Shape: {self.model.input_shape}")
                print(f"   - Output Shape: {self.model.output_shape}")
                print(f"   - Parameters: {self.model.count_params():,}")
                
            else:
                print(f"⚠️ Model path not found: {self.model_path}")
                print("🔧 Using default FER-2013 architecture...")
                self.create_default_model()
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("🔧 Falling back to default model...")
            self.create_default_model()
    
    def create_default_model(self):
        """Create default FER-2013 CNN architecture"""
        try:
            print("🏗️ Creating default FER-2013 CNN model...")
            
            # Simple CNN architecture untuk FER-2013
            self.model = keras.Sequential([
                # Input layer
                keras.layers.Input(shape=(48, 48, 1)),
                
                # Convolutional layers
                keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Dropout(0.25),
                
                keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Dropout(0.25),
                
                keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Dropout(0.25),
                
                # Flatten and dense layers
                keras.layers.Flatten(),
                keras.layers.Dense(256, activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.5),
                
                keras.layers.Dense(128, activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.5),
                
                # Output layer (7 emotions)
                keras.layers.Dense(7, activation='softmax')
            ])
            
            # Compile model
            self.model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            self.model_loaded = True
            print("✅ Default model created successfully!")
            print("⚠️ Note: This is untrained model. Please load your trained weights!")
            
        except Exception as e:
            print(f"❌ Error creating default model: {e}")
            self.model_loaded = False
    
    def preprocess_face(self, face_crop: np.ndarray) -> np.ndarray:
        """
        Preprocess face crop untuk input model
        """
        try:
            # Convert to grayscale if needed
            if len(face_crop.shape) == 3:
                face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            else:
                face_gray = face_crop
            
            # Resize to model input size
            face_resized = cv2.resize(face_gray, self.input_size)
            
            # Normalize pixel values (0-1)
            face_normalized = face_resized.astype(np.float32) / 255.0
            
            # Add batch and channel dimensions
            face_batch = np.expand_dims(face_normalized, axis=[0, -1])
            
            return face_batch
            
        except Exception as e:
            print(f"❌ Error preprocessing face: {e}")
            return None
    
    def predict_emotion(self, face_crop: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """
        Predict emotion menggunakan trained model
        """
        if not self.model_loaded or self.model is None:
            return 'Neutral', 0.5, np.array([])
        
        try:
            start_time = datetime.now()
            
            # Preprocess face
            face_input = self.preprocess_face(face_crop)
            if face_input is None:
                return 'Neutral', 0.5, np.array([])
            
            # Model prediction
            predictions = self.model.predict(face_input, verbose=0)
            prediction_probs = predictions[0]
            
            # Get predicted emotion
            predicted_class = np.argmax(prediction_probs)
            predicted_emotion = self.emotion_labels.get(predicted_class, 'Unknown')
            confidence = float(prediction_probs[predicted_class])
            
            # Calculate inference time
            inference_time = (datetime.now() - start_time).total_seconds()
            self.inference_times.append(inference_time)
            
            # Keep only last 100 inference times
            if len(self.inference_times) > 100:
                self.inference_times = self.inference_times[-100:]
            
            return predicted_emotion, confidence, prediction_probs
            
        except Exception as e:
            print(f"❌ Error in emotion prediction: {e}")
            return 'Neutral', 0.5, np.array([])
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        if not self.model_loaded:
            return {
                'status': 'not_loaded',
                'model_type': self.model_type,
                'input_size': self.input_size
            }
        
        try:
            # Model statistics
            total_params = self.model.count_params()
            trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
            non_trainable_params = total_params - trainable_params
            
            # Performance statistics
            avg_inference_time = np.mean(self.inference_times) if self.inference_times else 0
            recent_accuracy = np.mean(self.accuracy_history[-10:]) if self.accuracy_history else 0
            
            return {
                'status': 'loaded',
                'model_type': self.model_type,
                'model_path': self.model_path,
                'input_size': self.input_size,
                'architecture': {
                    'total_parameters': total_params,
                    'trainable_parameters': trainable_params,
                    'non_trainable_parameters': non_trainable_params,
                    'input_shape': self.model.input_shape,
                    'output_shape': self.model.output_shape
                },
                'performance': {
                    'avg_inference_time': avg_inference_time,
                    'recent_accuracy': recent_accuracy,
                    'total_predictions': len(self.inference_times)
                },
                'emotion_labels': self.emotion_labels
            }
            
        except Exception as e:
            print(f"❌ Error getting model info: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def update_accuracy(self, true_emotion: str, predicted_emotion: str, confidence: float):
        """Update accuracy history untuk model evaluation"""
        is_correct = true_emotion.lower() == predicted_emotion.lower()
        accuracy = 1.0 if is_correct else 0.0
        
        self.accuracy_history.append({
            'timestamp': datetime.now().isoformat(),
            'true_emotion': true_emotion,
            'predicted_emotion': predicted_emotion,
            'confidence': confidence,
            'is_correct': is_correct,
            'accuracy': accuracy
        })
        
        # Keep only last 1000 accuracy records
        if len(self.accuracy_history) > 1000:
            self.accuracy_history = self.accuracy_history[-1000:]
    
    def get_accuracy_stats(self) -> Dict:
        """Get accuracy statistics"""
        if not self.accuracy_history:
            return {
                'total_predictions': 0,
                'overall_accuracy': 0.0,
                'emotion_accuracy': {},
                'confidence_correlation': 0.0
            }
        
        try:
            total_predictions = len(self.accuracy_history)
            correct_predictions = sum([1 for record in self.accuracy_history if record['is_correct']])
            overall_accuracy = correct_predictions / total_predictions
            
            # Per-emotion accuracy
            emotion_counts = {}
            emotion_correct = {}
            
            for record in self.accuracy_history:
                emotion = record['predicted_emotion']
                if emotion not in emotion_counts:
                    emotion_counts[emotion] = 0
                    emotion_correct[emotion] = 0
                
                emotion_counts[emotion] += 1
                if record['is_correct']:
                    emotion_correct[emotion] += 1
            
            emotion_accuracy = {}
            for emotion in emotion_counts:
                emotion_accuracy[emotion] = emotion_correct[emotion] / emotion_counts[emotion]
            
            # Confidence correlation
            confidences = [record['confidence'] for record in self.accuracy_history]
            accuracies = [record['accuracy'] for record in self.accuracy_history]
            
            if len(confidences) > 1:
                confidence_correlation = np.corrcoef(confidences, accuracies)[0, 1]
            else:
                confidence_correlation = 0.0
            
            return {
                'total_predictions': total_predictions,
                'overall_accuracy': overall_accuracy,
                'emotion_accuracy': emotion_accuracy,
                'confidence_correlation': confidence_correlation,
                'recent_accuracy': np.mean([record['accuracy'] for record in self.accuracy_history[-100:]])
            }
            
        except Exception as e:
            print(f"❌ Error calculating accuracy stats: {e}")
            return {
                'total_predictions': 0,
                'overall_accuracy': 0.0,
                'emotion_accuracy': {},
                'confidence_correlation': 0.0
            }
    
    def save_model_info(self, filepath: str = "trained_model_info.json"):
        """Save model information dan performance stats"""
        try:
            model_info = self.get_model_info()
            accuracy_stats = self.get_accuracy_stats()
            
            save_data = {
                'model_info': model_info,
                'accuracy_stats': accuracy_stats,
                'timestamp': datetime.now().isoformat(),
                'total_inference_times': len(self.inference_times),
                'avg_inference_time': np.mean(self.inference_times) if self.inference_times else 0
            }
            
            with open(filepath, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            print(f"✅ Model info saved to: {filepath}")
            
        except Exception as e:
            print(f"❌ Error saving model info: {e}")
    
    def compare_with_rule_based(self, face_crop: np.ndarray, rule_based_result: Tuple[str, float]) -> Dict:
        """
        Compare trained model prediction dengan rule-based approach
        """
        if not self.model_loaded:
            return {
                'comparison': 'not_available',
                'rule_based': rule_based_result,
                'trained_model': None,
                'agreement': False
            }
        
        try:
            # Get trained model prediction
            trained_emotion, trained_confidence, trained_probs = self.predict_emotion(face_crop)
            
            rule_emotion, rule_confidence = rule_based_result
            
            # Check agreement
            agreement = rule_emotion.lower() == trained_emotion.lower()
            
            # Calculate confidence difference
            confidence_diff = abs(trained_confidence - rule_confidence)
            
            return {
                'comparison': 'available',
                'rule_based': {
                    'emotion': rule_emotion,
                    'confidence': rule_confidence
                },
                'trained_model': {
                    'emotion': trained_emotion,
                    'confidence': trained_confidence,
                    'probabilities': trained_probs.tolist()
                },
                'agreement': agreement,
                'confidence_difference': confidence_diff,
                'recommendation': 'trained_model' if trained_confidence > rule_confidence else 'rule_based'
            }
            
        except Exception as e:
            print(f"❌ Error in comparison: {e}")
            return {
                'comparison': 'error',
                'error': str(e)
            } 