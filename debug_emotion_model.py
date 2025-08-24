#!/usr/bin/env python3
"""
Debug script untuk menganalisis model emosi dan outputnya
"""
import torch
import torch.nn as nn
import cv2
import numpy as np
import os
from src.efficientnet_bilstm_model import EfficientNetBiLSTM
# from src.emotion_detector import preprocess_face_for_emotion
import traceback
from torchvision import transforms

# Define preprocessing for emotion detection
def preprocess_face_for_emotion(face_crop):
    """
    Preprocess face crop for emotion detection model
    """
    try:
        # Resize to 48x48 (typical for emotion models)
        face_resized = cv2.resize(face_crop, (48, 48))
        
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        
        # Convert to grayscale if needed (many emotion models use grayscale)
        # face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        # face_rgb = cv2.cvtColor(face_gray, cv2.COLOR_GRAY2RGB)
        
        # Convert to tensor and normalize
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            # Standard normalization for emotion models
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        face_tensor = transform(face_rgb)
        return face_tensor
        
    except Exception as e:
        print(f"Error in preprocess_face_for_emotion: {e}")
        # Return dummy tensor if preprocessing fails
        return torch.zeros(3, 48, 48)

def debug_model_architecture():
    """Debug arsitektur model"""
    print("=" * 60)
    print("🔍 DEBUGGING MODEL ARCHITECTURE")
    print("=" * 60)
    
    try:
        # Create model instance
        num_classes = 7
        model = EfficientNetBiLSTM(num_emotion_classes=num_classes)
        
        print(f"✅ Model created successfully")
        print(f"📊 Number of emotion classes: {num_classes}")
        print(f"🏗️ Model architecture:")
        print(model)
        
        # Check model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"📈 Total parameters: {total_params:,}")
        print(f"🎯 Trainable parameters: {trainable_params:,}")
        
        return model
        
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        traceback.print_exc()
        return None

def debug_model_loading():
    """Debug loading model weights"""
    print("\n" + "=" * 60)
    print("🔍 DEBUGGING MODEL LOADING")
    print("=" * 60)
    
    model_path = "models/emotion_efficientnet_bilstm_trained.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return None, None
    
    print(f"✅ Model file found: {model_path}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        print(f"✅ Checkpoint loaded successfully")
        print(f"📋 Checkpoint type: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict):
            print(f"🔑 Checkpoint keys: {list(checkpoint.keys())}")
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print(f"📦 Using 'model_state_dict' key")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print(f"📦 Using 'state_dict' key")
            else:
                state_dict = checkpoint
                print(f"📦 Using checkpoint directly as state_dict")
        else:
            state_dict = checkpoint
            print(f"📦 Checkpoint is direct state_dict")
        
        print(f"🔑 State dict keys:")
        for key in list(state_dict.keys())[:10]:  # Show first 10 keys
            print(f"  - {key}: {state_dict[key].shape if hasattr(state_dict[key], 'shape') else type(state_dict[key])}")
        if len(state_dict.keys()) > 10:
            print(f"  ... and {len(state_dict.keys()) - 10} more keys")
        
        return state_dict, checkpoint
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        traceback.print_exc()
        return None, None

def debug_model_compatibility(model, state_dict):
    """Debug compatibility between model and checkpoint"""
    print("\n" + "=" * 60)
    print("🔍 DEBUGGING MODEL COMPATIBILITY")
    print("=" * 60)
    
    if model is None or state_dict is None:
        print("❌ Cannot debug compatibility - model or state_dict is None")
        return False
    
    try:
        model_dict = model.state_dict()
        print(f"🏗️ Model expects {len(model_dict)} parameters")
        print(f"📦 Checkpoint has {len(state_dict)} parameters")
        
        # Check matching keys
        matching_keys = []
        missing_keys = []
        unexpected_keys = []
        shape_mismatches = []
        
        for key in model_dict.keys():
            if key in state_dict:
                if model_dict[key].shape == state_dict[key].shape:
                    matching_keys.append(key)
                else:
                    shape_mismatches.append((key, model_dict[key].shape, state_dict[key].shape))
            else:
                missing_keys.append(key)
        
        for key in state_dict.keys():
            if key not in model_dict:
                unexpected_keys.append(key)
        
        print(f"✅ Matching keys: {len(matching_keys)}")
        print(f"❌ Missing keys: {len(missing_keys)}")
        print(f"⚠️ Unexpected keys: {len(unexpected_keys)}")
        print(f"🔴 Shape mismatches: {len(shape_mismatches)}")
        
        if missing_keys:
            print(f"\n❌ Missing keys in checkpoint:")
            for key in missing_keys[:5]:  # Show first 5
                print(f"  - {key}")
            if len(missing_keys) > 5:
                print(f"  ... and {len(missing_keys) - 5} more")
        
        if unexpected_keys:
            print(f"\n⚠️ Unexpected keys in checkpoint:")
            for key in unexpected_keys[:5]:  # Show first 5
                print(f"  - {key}")
            if len(unexpected_keys) > 5:
                print(f"  ... and {len(unexpected_keys) - 5} more")
        
        if shape_mismatches:
            print(f"\n🔴 Shape mismatches:")
            for key, expected, actual in shape_mismatches:
                print(f"  - {key}: expected {expected}, got {actual}")
        
        # Try to load compatible weights
        print(f"\n🔧 Attempting to load compatible weights...")
        compatible_dict = {k: v for k, v in state_dict.items() 
                          if k in model_dict and model_dict[k].shape == v.shape}
        
        model.load_state_dict(compatible_dict, strict=False)
        print(f"✅ Loaded {len(compatible_dict)} compatible parameters")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in compatibility check: {e}")
        traceback.print_exc()
        return False

def debug_model_inference():
    """Debug model inference dengan input dummy"""
    print("\n" + "=" * 60)
    print("🔍 DEBUGGING MODEL INFERENCE")
    print("=" * 60)
    
    try:
        # Create model
        model = EfficientNetBiLSTM(num_emotion_classes=7)
        
        # Load weights
        state_dict, _ = debug_model_loading()
        if state_dict:
            compatible_dict = {k: v for k, v in state_dict.items() 
                              if k in model.state_dict() and model.state_dict()[k].shape == v.shape}
            model.load_state_dict(compatible_dict, strict=False)
        
        model.eval()
        
        # Create dummy input: (batch_size=1, sequence_length=5, channels=3, height=48, width=48)
        batch_size = 1
        sequence_length = 5
        channels = 3
        height = 48
        width = 48
        
        dummy_input = torch.randn(batch_size, sequence_length, channels, height, width)
        print(f"🎯 Dummy input shape: {dummy_input.shape}")
        
        # Forward pass
        with torch.no_grad():
            output = model(dummy_input)
            print(f"📊 Raw output shape: {output.shape}")
            print(f"📊 Raw output values: {output}")
            
            # Apply softmax
            probabilities = torch.softmax(output, dim=1)
            print(f"📊 Probabilities shape: {probabilities.shape}")
            print(f"📊 Probabilities: {probabilities}")
            
            # Get prediction
            max_conf, predicted_idx = torch.max(probabilities, 1)
            print(f"🎯 Predicted class index: {predicted_idx.item()}")
            print(f"🎯 Confidence: {max_conf.item():.4f}")
            
            # Emotion labels
            emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprised"]
            if predicted_idx.item() < len(emotion_labels):
                predicted_emotion = emotion_labels[predicted_idx.item()]
                print(f"😊 Predicted emotion: {predicted_emotion}")
            
            return True
            
    except Exception as e:
        print(f"❌ Error in model inference: {e}")
        traceback.print_exc()
        return False

def debug_real_image_processing():
    """Debug processing dengan gambar asli"""
    print("\n" + "=" * 60)
    print("🔍 DEBUGGING REAL IMAGE PROCESSING")
    print("=" * 60)
    
    # Cari gambar test di known_faces
    test_images = []
    for root, dirs, files in os.walk("known_faces"):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                test_images.append(os.path.join(root, file))
    
    if not test_images:
        print("❌ No test images found in known_faces directory")
        return False
    
    print(f"✅ Found {len(test_images)} test images")
    
    try:
        # Load model
        model = EfficientNetBiLSTM(num_emotion_classes=7)
        state_dict, _ = debug_model_loading()
        if state_dict:
            compatible_dict = {k: v for k, v in state_dict.items() 
                              if k in model.state_dict() and model.state_dict()[k].shape == v.shape}
            model.load_state_dict(compatible_dict, strict=False)
        
        model.eval()
        
        for i, img_path in enumerate(test_images[:3]):  # Test first 3 images
            print(f"\n📷 Testing image {i+1}: {img_path}")
            
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                print(f"❌ Failed to load image: {img_path}")
                continue
            
            print(f"📊 Original image shape: {image.shape}")
            
            # Preprocess face (simulate face detection result)
            # For demo, use the whole image and resize
            face_roi = cv2.resize(image, (48, 48))
            
            # Preprocess for emotion detection
            processed_face = preprocess_face_for_emotion(face_roi)
            print(f"📊 Processed face shape: {processed_face.shape}")
            
            # Create sequence (repeat same frame 5 times for demo)
            sequence = torch.stack([processed_face] * 5).unsqueeze(0)  # (1, 5, C, H, W)
            print(f"📊 Sequence shape: {sequence.shape}")
            
            # Inference
            with torch.no_grad():
                output = model(sequence)
                probabilities = torch.softmax(output, dim=1)
                max_conf, predicted_idx = torch.max(probabilities, 1)
                
                emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprised"]
                predicted_emotion = emotion_labels[predicted_idx.item()]
                
                print(f"🎯 Predicted emotion: {predicted_emotion}")
                print(f"🎯 Confidence: {max_conf.item():.4f}")
                print(f"📊 All probabilities:")
                for j, (emotion, prob) in enumerate(zip(emotion_labels, probabilities[0])):
                    print(f"  {emotion}: {prob.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in real image processing: {e}")
        traceback.print_exc()
        return False

def main():
    """Main debug function"""
    print("🚀 STARTING EMOTION MODEL DEBUG")
    print("=" * 60)
    
    # 1. Debug model architecture
    model = debug_model_architecture()
    
    # 2. Debug model loading
    state_dict, checkpoint = debug_model_loading()
    
    # 3. Debug compatibility
    compatibility_ok = debug_model_compatibility(model, state_dict)
    
    # 4. Debug inference
    inference_ok = debug_model_inference()
    
    # 5. Debug real image processing
    real_processing_ok = debug_real_image_processing()
    
    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)
    print(f"Model Architecture: {'✅' if model is not None else '❌'}")
    print(f"Model Loading: {'✅' if state_dict is not None else '❌'}")
    print(f"Compatibility: {'✅' if compatibility_ok else '❌'}")
    print(f"Inference: {'✅' if inference_ok else '❌'}")
    print(f"Real Image Processing: {'✅' if real_processing_ok else '❌'}")

if __name__ == "__main__":
    main()