#!/usr/bin/env python3
"""
Test improved emotion detection system with bias correction
"""
import torch
import cv2
import numpy as np
import os
from src.efficientnet_bilstm_model import EfficientNetBiLSTM
from src.emotion_detector import analyze_face_expression_simple
from torchvision import transforms

def load_model():
    """Load the fixed model"""
    model = EfficientNetBiLSTM(num_emotion_classes=7)
    checkpoint = torch.load("models/emotion_efficientnet_bilstm_trained.pth", map_location='cpu')
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model

def test_improved_emotion_detection(image_path, model):
    """Test improved emotion detection with bias correction"""
    print(f"\nTesting improved system for: {image_path}")
    print("=" * 60)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load: {image_path}")
        return
    
    # Test simple face analysis first
    print("\n1. Simple Face Analysis:")
    simple_emotion, simple_conf = analyze_face_expression_simple(image)
    print(f"   Result: {simple_emotion} (confidence: {simple_conf:.3f})")
    
    # Test model prediction
    print("\n2. Model Prediction:")
    face_48 = cv2.resize(image, (48, 48))
    face_rgb = cv2.cvtColor(face_48, cv2.COLOR_BGR2RGB)
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    face_tensor = transform(face_rgb)
    sequence = torch.stack([face_tensor] * 5).unsqueeze(0)
    
    with torch.no_grad():
        output = model(sequence)
        probabilities = torch.softmax(output, dim=1)
        max_conf, predicted_idx = torch.max(probabilities, 1)
        
        emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprised"]
        predicted_emotion = emotion_labels[predicted_idx.item()]
        emotion_conf = max_conf.item()
        
        print(f"   Raw prediction: {predicted_emotion} (confidence: {emotion_conf:.3f})")
        
        # Show all probabilities
        all_probs = probabilities[0].tolist()
        for i, (emotion, prob) in enumerate(zip(emotion_labels, all_probs)):
            print(f"     {emotion}: {prob:.4f}")
    
    # Test bias correction logic
    print("\n3. Bias Correction Logic:")
    
    # Check for model bias
    if predicted_emotion == "Angry" and emotion_conf > 0.45:
        # Check if Happy is the second highest
        sorted_probs = sorted(enumerate(all_probs), key=lambda x: x[1], reverse=True)
        second_emotion_idx = sorted_probs[1][0]
        second_emotion = emotion_labels[second_emotion_idx]
        second_conf = sorted_probs[1][1]
        
        print(f"   Model bias detected: Angry ({emotion_conf:.3f}) vs {second_emotion} ({second_conf:.3f})")
        
        # If Happy is second and close to Angry, prefer Happy for "happy-looking" faces
        if second_emotion == "Happy" and second_conf > 0.3:
            print(f"   Simple analysis suggests: {simple_emotion} ({simple_conf:.3f})")
            
            if simple_emotion == "Happy":
                final_emotion = "Happy"
                final_conf = (second_conf + simple_conf) / 2
                print(f"   🎯 FINAL RESULT: {final_emotion} (confidence: {final_conf:.3f})")
                print(f"   ✅ CORRECTED from Angry to Happy!")
            else:
                final_emotion = predicted_emotion
                final_conf = emotion_conf
                print(f"   🎯 FINAL RESULT: {final_emotion} (confidence: {final_conf:.3f})")
                print(f"   ⚠️ Keeping Angry as simple analysis doesn't confirm Happy")
        else:
            final_emotion = predicted_emotion
            final_conf = emotion_conf
            print(f"   🎯 FINAL RESULT: {final_emotion} (confidence: {final_conf:.3f})")
    else:
        final_emotion = predicted_emotion
        final_conf = emotion_conf
        print(f"   🎯 FINAL RESULT: {final_emotion} (confidence: {final_conf:.3f})")
        print(f"   ✅ No bias correction needed")

def main():
    print("Testing Improved Emotion Detection System")
    print("=" * 60)
    
    # Load model
    model = load_model()
    
    # Test with Nesya's image
    nesya_path = "known_faces/nesya/nesya.png"
    if os.path.exists(nesya_path):
        test_improved_emotion_detection(nesya_path, model)
    
    # Test with other images
    test_images = [
        "known_faces/fadel/fadel.jpg",
        "known_faces/oji/oji.png",
        "known_faces/prabowo/prabowo.png"
    ]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            test_improved_emotion_detection(img_path, model)

if __name__ == "__main__":
    main() 