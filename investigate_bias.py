#!/usr/bin/env python3
"""
Investigate model bias and test different preprocessing approaches
"""
import torch
import cv2
import numpy as np
import os
from src.efficientnet_bilstm_model import EfficientNetBiLSTM
from torchvision import transforms
import matplotlib.pyplot as plt

def load_model():
    """Load the fixed model"""
    model = EfficientNetBiLSTM(num_emotion_classes=7)
    checkpoint = torch.load("models/emotion_efficientnet_bilstm_trained.pth", map_location='cpu')
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model

def test_different_preprocessing(image_path, model):
    """Test different preprocessing approaches"""
    print(f"\nTesting different preprocessing for: {image_path}")
    print("=" * 60)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load: {image_path}")
        return
    
    emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprised"]
    
    # Test 1: Current preprocessing (48x48)
    print("\nTest 1: Current preprocessing (48x48)")
    face_48 = cv2.resize(image, (48, 48))
    face_rgb_48 = cv2.cvtColor(face_48, cv2.COLOR_BGR2RGB)
    
    transform_48 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    face_tensor_48 = transform_48(face_rgb_48)
    sequence_48 = torch.stack([face_tensor_48] * 5).unsqueeze(0)
    
    with torch.no_grad():
        output_48 = model(sequence_48)
        probs_48 = torch.softmax(output_48, dim=1)
        max_conf_48, pred_idx_48 = torch.max(probs_48, 1)
        
        print(f"Predicted: {emotion_labels[pred_idx_48.item()]} ({max_conf_48.item():.4f})")
        for i, (emotion, prob) in enumerate(zip(emotion_labels, probs_48[0])):
            print(f"  {emotion}: {prob:.4f}")
    
    # Test 2: Larger size (224x224 - ImageNet standard)
    print("\nTest 2: Larger size (224x224)")
    face_224 = cv2.resize(image, (224, 224))
    face_rgb_224 = cv2.cvtColor(face_224, cv2.COLOR_BGR2RGB)
    
    transform_224 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    face_tensor_224 = transform_224(face_rgb_224)
    # Resize to 48x48 for model input
    face_tensor_224_resized = torch.nn.functional.interpolate(
        face_tensor_224.unsqueeze(0), size=(48, 48), mode='bilinear', align_corners=False
    ).squeeze(0)
    
    sequence_224 = torch.stack([face_tensor_224_resized] * 5).unsqueeze(0)
    
    with torch.no_grad():
        output_224 = model(sequence_224)
        probs_224 = torch.softmax(output_224, dim=1)
        max_conf_224, pred_idx_224 = torch.max(probs_224, 1)
        
        print(f"Predicted: {emotion_labels[pred_idx_224.item()]} ({max_conf_224.item():.4f})")
        for i, (emotion, prob) in enumerate(zip(emotion_labels, probs_224[0])):
            print(f"  {emotion}: {prob:.4f}")
    
    # Test 3: Grayscale preprocessing
    print("\nTest 3: Grayscale preprocessing")
    face_gray = cv2.cvtColor(cv2.resize(image, (48, 48)), cv2.COLOR_BGR2GRAY)
    face_gray_rgb = cv2.cvtColor(face_gray, cv2.COLOR_GRAY2RGB)
    
    face_tensor_gray = transform_48(face_gray_rgb)
    sequence_gray = torch.stack([face_tensor_gray] * 5).unsqueeze(0)
    
    with torch.no_grad():
        output_gray = model(sequence_gray)
        probs_gray = torch.softmax(output_gray, dim=1)
        max_conf_gray, pred_idx_gray = torch.max(probs_gray, 1)
        
        print(f"Predicted: {emotion_labels[pred_idx_gray.item()]} ({max_conf_gray.item():.4f})")
        for i, (emotion, prob) in enumerate(zip(emotion_labels, probs_gray[0])):
            print(f"  {emotion}: {prob:.4f}")
    
    # Test 4: Different normalization
    print("\nTest 4: Different normalization (0-1 range)")
    face_rgb_48 = cv2.cvtColor(face_48, cv2.COLOR_BGR2RGB)
    
    transform_01 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),  # This normalizes to 0-1
        # No ImageNet normalization
    ])
    
    face_tensor_01 = transform_01(face_rgb_48)
    sequence_01 = torch.stack([face_tensor_01] * 5).unsqueeze(0)
    
    with torch.no_grad():
        output_01 = model(sequence_01)
        probs_01 = torch.softmax(output_01, dim=1)
        max_conf_01, pred_idx_01 = torch.max(probs_01, 1)
        
        print(f"Predicted: {emotion_labels[pred_idx_01.item()]} ({max_conf_01.item():.4f})")
        for i, (emotion, prob) in enumerate(zip(emotion_labels, probs_01[0])):
            print(f"  {emotion}: {prob:.4f}")

def analyze_model_bias(model):
    """Analyze model bias with random inputs"""
    print("\nAnalyzing Model Bias with Random Inputs")
    print("=" * 60)
    
    emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprised"]
    predictions = {emotion: 0 for emotion in emotion_labels}
    total_tests = 10
    
    for i in range(total_tests):
        # Generate random face-like input
        random_input = torch.randn(1, 5, 3, 48, 48)
        
        with torch.no_grad():
            output = model(random_input)
            probs = torch.softmax(output, dim=1)
            _, pred_idx = torch.max(probs, 1)
            
            predicted_emotion = emotion_labels[pred_idx.item()]
            predictions[predicted_emotion] += 1
    
    print(f"Random input predictions (over {total_tests} tests):")
    for emotion, count in predictions.items():
        percentage = (count / total_tests) * 100
        print(f"  {emotion}: {count}/{total_tests} ({percentage:.1f}%)")

def test_face_detection_quality(image_path):
    """Test if face detection is working properly"""
    print(f"\nTesting face detection quality for: {image_path}")
    print("=" * 60)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load: {image_path}")
        return
    
    print(f"Original image shape: {image.shape}")
    
    # Try to detect faces using OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    print(f"Detected faces: {len(faces)}")
    
    if len(faces) > 0:
        # Use the largest face
        largest_face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = largest_face
        
        # Crop face
        face_crop = image[y:y+h, x:x+w]
        print(f"Face crop shape: {face_crop.shape}")
        
        # Test with face crop
        face_resized = cv2.resize(face_crop, (48, 48))
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        face_tensor = transform(face_rgb)
        sequence = torch.stack([face_tensor] * 5).unsqueeze(0)
        
        model = load_model()
        with torch.no_grad():
            output = model(sequence)
            probs = torch.softmax(output, dim=1)
            max_conf, pred_idx = torch.max(probs, 1)
            
            emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprised"]
            predicted_emotion = emotion_labels[pred_idx.item()]
            
            print(f"Face crop prediction: {predicted_emotion} ({max_conf.item():.4f})")
            for i, (emotion, prob) in enumerate(zip(emotion_labels, probs[0])):
                print(f"  {emotion}: {prob:.4f}")

def main():
    print("INVESTIGATING MODEL BIAS")
    print("=" * 60)
    
    # Load model
    model = load_model()
    
    # Test with Nesya's image
    nesya_path = "known_faces/nesya/nesya.png"
    if os.path.exists(nesya_path):
        test_different_preprocessing(nesya_path, model)
        test_face_detection_quality(nesya_path)
    
    # Analyze model bias
    analyze_model_bias(model)
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    print("1. Check if model was trained with proper emotion labels")
    print("2. Verify preprocessing matches training data")
    print("3. Consider retraining with balanced dataset")
    print("4. Test with different face detection methods")
    print("5. Consider using emotion-specific preprocessing")

if __name__ == "__main__":
    main() 