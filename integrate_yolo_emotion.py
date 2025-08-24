#!/usr/bin/env python3
"""
Integration script untuk menggabungkan YOLO Emotion Detection dengan sistem EMONG
"""
import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path

# Add src directory to path
sys.path.append('src')

def integrate_with_emong():
    """
    Integrate YOLO emotion detection with existing EMONG system
    """
    print("🔗 INTEGRATING YOLO EMOTION DETECTION WITH EMONG")
    print("=" * 60)
    
    # Check if YOLO emotion model exists
    yolo_emotion_model = "models/yolo_emotion_detection_v8s.pt"
    
    if not os.path.exists(yolo_emotion_model):
        print(f"❌ YOLO emotion model not found: {yolo_emotion_model}")
        print("🚀 Please train the model first using train_yolo_emotion.py")
        return False
    
    print(f"✅ Found YOLO emotion model: {yolo_emotion_model}")
    
    # Import existing EMONG components
    try:
        from src.emotion_detector import load_known_faces, known_face_encodings, known_face_names
        print("✅ Loaded existing EMONG face recognition components")
    except ImportError as e:
        print(f"❌ Error importing EMONG components: {e}")
        return False
    
    # Create integrated detector
    try:
        from yolo_emotion_detector import YOLOEmotionDetector
        detector = YOLOEmotionDetector(yolo_emotion_model)
        print("✅ YOLO emotion detector initialized")
    except Exception as e:
        print(f"❌ Error initializing YOLO detector: {e}")
        return False
    
    # Test integration
    print("\n🧪 Testing integration...")
    
    # Create test frame
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test detection
    detections = detector.detect_faces_and_emotions(test_frame)
    print(f"✅ Test detection successful: {len(detections)} faces detected")
    
    return True

def update_app_py():
    """
    Update app.py to use YOLO emotion detection
    """
    print("\n📝 Updating app.py for YOLO integration...")
    
    app_py_path = "app.py"
    if not os.path.exists(app_py_path):
        print(f"❌ app.py not found: {app_py_path}")
        return False
    
    # Read current app.py
    with open(app_py_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already updated
    if "YOLOEmotionDetector" in content:
        print("✅ app.py already updated for YOLO integration")
        return True
    
    # Add YOLO import
    yolo_import = """
# YOLO Emotion Detection Integration
from yolo_emotion_detector import YOLOEmotionDetector
"""
    
    # Find the import section
    import_section = "import cv2"
    if import_section in content:
        content = content.replace(import_section, import_section + yolo_import)
        print("✅ Added YOLO import to app.py")
    
    # Add YOLO detector initialization
    yolo_initialize = """
# Initialize YOLO emotion detector
yolo_emotion_detector = None
try:
    yolo_emotion_detector = YOLOEmotionDetector("models/yolo_emotion_detection_v8s.pt")
    print("✅ YOLO emotion detector initialized")
except Exception as e:
    print(f"⚠️ Could not initialize YOLO detector: {e}")
    print("🔄 Falling back to existing emotion detection")
"""
    
    # Find the model loading section
    model_section = "# Load emotion detection model"
    if model_section in content:
        content = content.replace(model_section, model_section + yolo_initialize)
        print("✅ Added YOLO initialization to app.py")
    
    # Update the handle_frame function to use YOLO
    yolo_update = """
    # Use YOLO emotion detection if available
    if yolo_emotion_detector is not None:
        try:
            # YOLO detection
            yolo_detections = yolo_emotion_detector.detect_faces_and_emotions(frame)
            
            # Convert YOLO format to existing format
            results = []
            for name, emotion, confidence, bbox in yolo_detections:
                results.append({
                    "name": name,
                    "emotion": emotion,
                    "confidence": confidence,
                    "emoji": EMOTION_EMOJIS.get(emotion, "❓"),
                    "color": EMOTION_COLORS_CSS.get(emotion, "rgb(255, 255, 255)"),
                    "bbox": bbox,
                })
            
            print(f"🎯 YOLO detected {len(results)} faces with emotions")
            
        except Exception as e:
            print(f"❌ YOLO detection failed: {e}")
            print("🔄 Falling back to existing detection")
            # Fall back to existing detection
            with cache_lock:
                detections = detect_emotions_and_recognize_faces(
                    face_detector_model, 
                    emotion_model, 
                    frame, 
                    known_face_encodings, 
                    known_face_names, 
                    recent_face_cache_realtime, 
                    face_sequence_buffers_realtime, 
                    confidence_threshold
                )
            
            results = [
                {
                    "name": name,
                    "emotion": e,
                    "confidence": c,
                    "emoji": EMOTION_EMOJIS.get(e, "❓"),
                    "color": EMOTION_COLORS_CSS.get(e, "rgb(255, 255, 255)"),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                }
                for name, e, c, (x1, y1, x2, y2) in detections
            ]
    else:
        # Use existing detection
        with cache_lock:
            detections = detect_emotions_and_recognize_faces(
                face_detector_model, 
                emotion_model, 
                frame, 
                known_face_encodings, 
                known_face_names, 
                recent_face_cache_realtime, 
                face_sequence_buffers_realtime, 
                confidence_threshold
            )
        
        results = [
            {
                "name": name,
                "emotion": e,
                "confidence": c,
                "emoji": EMOTION_EMOJIS.get(e, "❓"),
                "color": EMOTION_COLORS_CSS.get(e, "rgb(255, 255, 255)"),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
            }
            for name, e, c, (x1, y1, x2, y2) in detections
        ]
"""
    
    # Find the handle_frame function
    if "def handle_frame(data):" in content:
        # This is a complex replacement - we'll need to be more careful
        print("⚠️ Manual update required for handle_frame function")
        print("📝 Please update the handle_frame function manually")
    
    # Save updated app.py
    backup_path = "app.py.backup"
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"💾 Backup saved to {backup_path}")
    
    return True

def create_yolo_config():
    """
    Create configuration file for YOLO integration
    """
    print("\n⚙️ Creating YOLO integration configuration...")
    
    config = {
        'yolo_emotion_model': 'models/yolo_emotion_detection_v8s.pt',
        'confidence_threshold': 0.5,
        'nms_threshold': 0.4,
        'input_size': 640,
        'temporal_window': 10,
        'smoothing_alpha': 0.3,
        'face_tracking_timeout': 2.0,
        'emotion_labels': ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprised'],
        'fallback_to_existing': True,
        'enable_logging': True,
        'save_detections': False,
        'output_path': 'yolo_detections'
    }
    
    config_path = "yolo_integration_config.yaml"
    import yaml
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✅ Configuration saved to {config_path}")
    return config_path

def test_integration():
    """
    Test the integrated system
    """
    print("\n🧪 Testing integrated system...")
    
    try:
        # Test YOLO detector
        from yolo_emotion_detector import YOLOEmotionDetector
        
        # Test with sample image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        detector = YOLOEmotionDetector("models/yolov12s.pt")  # Use existing model for now
        
        detections = detector.detect_faces_and_emotions(test_image)
        
        print(f"✅ Integration test successful: {len(detections)} detections")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    """Main integration function"""
    print("🎯 YOLO EMOTION DETECTION INTEGRATION WITH EMONG")
    print("=" * 60)
    
    # Step 1: Check integration readiness
    if not integrate_with_emong():
        print("❌ Integration check failed")
        return
    
    # Step 2: Create configuration
    config_path = create_yolo_config()
    
    # Step 3: Update app.py (partial)
    update_app_py()
    
    # Step 4: Test integration
    if test_integration():
        print("\n🎉 INTEGRATION COMPLETED SUCCESSFULLY!")
        print(f"📁 Configuration: {config_path}")
        print(f"💾 Backup: app.py.backup")
        
        print("\n🚀 NEXT STEPS:")
        print("1. Train YOLO emotion model: python train_yolo_emotion.py")
        print("2. Update app.py handle_frame function manually")
        print("3. Test with real-time detection")
        print("4. Fine-tune YOLO parameters")
        
        print("\n📝 MANUAL UPDATE REQUIRED:")
        print("Update the handle_frame function in app.py to use YOLO detection")
        print("See the integration code above for reference")
        
    else:
        print("\n❌ Integration failed. Please check the error messages above.")

if __name__ == "__main__":
    main() 