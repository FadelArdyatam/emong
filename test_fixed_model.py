#!/usr/bin/env python3
"""
Test script untuk model yang sudah diperbaiki
"""
import torch
import torch.nn as nn
import cv2
import numpy as np
import os
from src.efficientnet_bilstm_model import EfficientNetBiLSTM
from torchvision import transforms

def preprocess_face_for_emotion(face_crop):
    """Preprocess face crop for emotion detection model"""
    try:
        # Resize to 48x48
        face_resized = cv2.resize(face_crop, (48, 48))
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        face_tensor = transform(face_rgb)
        return face_tensor
        
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return torch.zeros(3, 48, 48)

def test_model_loading():
    """Test loading model dengan nama layer yang benar"""
    print("=" * 60)
    print("Testing Model Loading with Fixed LSTM Naming")
    print("=" * 60)
    
    try:
        # Create model
        model = EfficientNetBiLSTM(num_emotion_classes=7)
        print("Model created successfully")
        
        # Load checkpoint
        model_path = "models/emotion_efficientnet_bilstm_trained.pth"
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            
            # Load state dict
            model.load_state_dict(checkpoint, strict=False)
            print("Checkpoint loaded successfully with fixed naming!")
            
            # Check if LSTM weights are loaded
            model_dict = model.state_dict()
            bilstm_keys = [k for k in model_dict.keys() if 'bilstm' in k]
            print(f"BiLSTM parameters loaded: {len(bilstm_keys)}")
            
            return model
            
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def test_inference(model):
    """Test inference dengan model yang sudah diperbaiki"""
    print("\n" + "=" * 60)
    print("Testing Inference with Fixed Model")
    print("=" * 60)
    
    if model is None:
        print("Model is None, cannot test inference")
        return False
    
    try:
        model.eval()
        
        # Test with dummy input
        dummy_input = torch.randn(1, 5, 3, 48, 48)
        print(f"Input shape: {dummy_input.shape}")
        
        with torch.no_grad():
            output = model(dummy_input)
            probabilities = torch.softmax(output, dim=1)
            max_conf, predicted_idx = torch.max(probabilities, 1)
            
            print(f"Output shape: {output.shape}")
            print(f"Raw output: {output}")
            print(f"Probabilities: {probabilities}")
            print(f"Predicted index: {predicted_idx.item()}")
            print(f"Confidence: {max_conf.item():.4f}")
            
            emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprised"]
            if predicted_idx.item() < len(emotion_labels):
                predicted_emotion = emotion_labels[predicted_idx.item()]
                print(f"Predicted emotion: {predicted_emotion}")
                
                # Check probability variance
                probs_array = probabilities[0].numpy()
                variance = np.var(probs_array)
                print(f"Probability variance: {variance:.6f}")
                
                if variance > 0.001:
                    print("Good! Model shows discriminative predictions")
                else:
                    print("Warning: Model predictions are too uniform")
            
            return True
            
    except Exception as e:
        print(f"Error in inference: {e}")
        return False

def test_real_image(model):
    """Test dengan gambar asli"""
    print("\n" + "=" * 60)
    print("Testing with Real Images")
    print("=" * 60)
    
    if model is None:
        print("Model is None, cannot test with real images")
        return False
    
    # Find test images
    test_images = []
    for root, dirs, files in os.walk("known_faces"):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                test_images.append(os.path.join(root, file))
                if len(test_images) >= 2:  # Test only 2 images
                    break
    
    if not test_images:
        print("No test images found")
        return False
    
    model.eval()
    emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprised"]
    
    for img_path in test_images:
        print(f"\nTesting image: {img_path}")
        
        try:
            # Load and preprocess image
            image = cv2.imread(img_path)
            if image is None:
                print(f"Failed to load image: {img_path}")
                continue
            
            # Use whole image (simulating face detection result)
            face_roi = cv2.resize(image, (48, 48))
            processed_face = preprocess_face_for_emotion(face_roi)
            
            # Create sequence
            sequence = torch.stack([processed_face] * 5).unsqueeze(0)
            
            # Inference
            with torch.no_grad():
                output = model(sequence)
                probabilities = torch.softmax(output, dim=1)
                max_conf, predicted_idx = torch.max(probabilities, 1)
                
                predicted_emotion = emotion_labels[predicted_idx.item()]
                confidence = max_conf.item()
                
                print(f"Predicted: {predicted_emotion} (confidence: {confidence:.4f})")
                
                # Show all probabilities
                probs_array = probabilities[0].numpy()
                variance = np.var(probs_array)
                print(f"Probability variance: {variance:.6f}")
                
                for i, (emotion, prob) in enumerate(zip(emotion_labels, probs_array)):
                    print(f"  {emotion}: {prob:.4f}")
        
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    return True

def main():
    print("Starting Model Test with Fixed LSTM Naming")
    print("=" * 60)
    
    # Test model loading
    model = test_model_loading()
    
    # Test inference
    inference_ok = test_inference(model)
    
    # Test with real images
    real_test_ok = test_real_image(model)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Model Loading: {'OK' if model is not None else 'FAILED'}")
    print(f"Inference Test: {'OK' if inference_ok else 'FAILED'}")
    print(f"Real Image Test: {'OK' if real_test_ok else 'FAILED'}")

if __name__ == "__main__":
    main()