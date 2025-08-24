#!/usr/bin/env python3
"""
Simple test untuk YOLO model
"""

import os
import sys

def check_model_file():
    """Check apakah model file ada dan valid"""
    
    model_path = "models/yolo_emotion_detection_v8s.pt"
    
    print(f"🔍 Checking model file: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file tidak ditemukan: {model_path}")
        return False
    
    file_size = os.path.getsize(model_path)
    print(f"✅ Model file ditemukan")
    print(f"📊 File size: {file_size} bytes ({file_size / (1024*1024):.2f} MB)")
    
    if file_size < 1000000:  # Less than 1MB
        print("⚠️ File terlalu kecil, mungkin corrupt")
        return False
    
    return True

def check_environment():
    """Check environment dan dependencies"""
    
    print("\n🔍 Checking environment...")
    
    # Check Python version
    print(f"🐍 Python version: {sys.version}")
    
    # Check if we can import basic modules
    try:
        import cv2
        print(f"✅ OpenCV version: {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV import error: {e}")
    
    try:
        import numpy as np
        print(f"✅ NumPy version: {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy import error: {e}")
    
    # Try to import YOLO
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics YOLO imported successfully")
        
        # Try to create YOLO instance
        model = YOLO('yolov8s.pt')  # Try default model first
        print("✅ Default YOLO model loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ YOLO import/load error: {e}")
        return False

def test_custom_model():
    """Test custom model jika environment OK"""
    
    print("\n🔍 Testing custom model...")
    
    try:
        from ultralytics import YOLO
        
        model_path = "models/yolo_emotion_detection_v8s.pt"
        
        print(f"🔄 Loading custom model: {model_path}")
        model = YOLO(model_path)
        
        print(f"✅ Custom model loaded successfully!")
        print(f"📊 Classes: {model.names}")
        print(f"🔢 Number of classes: {len(model.names)}")
        
        # Test dengan image sederhana
        import numpy as np
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        print("🔄 Running inference test...")
        results = model(test_image, conf=0.01)  # Very low confidence
        
        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                print(f"📦 Test detections: {len(boxes)}")
                
                if len(boxes) > 0:
                    print("✅ Model working! Found detections")
                    return True
                else:
                    print("⚠️ Model loaded but no detections")
                    return False
            else:
                print("⚠️ Model loaded but no boxes attribute")
                return False
        else:
            print("⚠️ Model loaded but no results")
            return False
            
    except Exception as e:
        print(f"❌ Custom model test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    
    print("🚀 YOLO Model Diagnostic Tool")
    print("=" * 50)
    
    # Check 1: Model file
    model_ok = check_model_file()
    
    # Check 2: Environment
    env_ok = check_environment()
    
    # Check 3: Custom model (if environment OK)
    if env_ok and model_ok:
        custom_ok = test_custom_model()
        
        if custom_ok:
            print("\n🎉 All tests passed! Model should work.")
        else:
            print("\n⚠️ Model loaded but not detecting properly.")
            print("   This might be a training data issue.")
    else:
        print("\n❌ Environment or model file issues detected.")
        print("   Please fix these before testing the model.")

if __name__ == "__main__":
    main() 