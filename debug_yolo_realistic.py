#!/usr/bin/env python3
"""
Debug script untuk test YOLO dengan image yang lebih realistis
"""

import cv2
import numpy as np
from ultralytics import YOLO
import torch

def test_with_realistic_images():
    """Test YOLO dengan image yang lebih realistis"""
    
    model_path = "models/yolo_emotion_detection_v8s.pt"
    
    print(f"🔄 Testing YOLO model: {model_path}")
    
    try:
        # Load model
        model = YOLO(model_path)
        print(f"✅ Model loaded successfully!")
        print(f"📊 Classes: {model.names}")
        print(f"🔢 Number of classes: {len(model.names)}")
        
        # Test 1: Image dengan wajah yang lebih realistis
        print("\n🔄 Test 1: Realistic face-like image...")
        
        # Buat image dengan wajah yang lebih detail
        face_image = np.random.randint(80, 180, (640, 640, 3), dtype=np.uint8)
        
        # Kepala (oval)
        cv2.ellipse(face_image, (320, 320), (150, 200), 0, 0, 360, (255, 255, 255), -1)
        
        # Mata (oval)
        cv2.ellipse(face_image, (280, 280), (25, 15), 0, 0, 360, (0, 0, 0), -1)
        cv2.ellipse(face_image, (360, 280), (25, 15), 0, 0, 360, (0, 0, 0), -1)
        
        # Hidung (segitiga)
        nose_points = np.array([[320, 320], [300, 350], [340, 350]], np.int32)
        cv2.fillPoly(face_image, [nose_points], (200, 200, 200))
        
        # Mulut (senyum)
        cv2.ellipse(face_image, (320, 380), (40, 20), 0, 0, 180, (0, 0, 0), 3)
        
        # Alis
        cv2.line(face_image, (260, 250), (300, 260), (0, 0, 0), 3)
        cv2.line(face_image, (340, 260), (380, 250), (0, 0, 0), 3)
        
        print(f"🖼️ Face image shape: {face_image.shape}")
        
        # Test dengan berbagai confidence threshold
        for conf_threshold in [0.01, 0.05, 0.1, 0.2, 0.3]:
            print(f"\n🔍 Testing with confidence threshold: {conf_threshold}")
            results = model(face_image, conf=conf_threshold)
            
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    print(f"  📦 Detections: {len(boxes)}")
                    
                    for i, box in enumerate(boxes):
                        if hasattr(box, 'xyxy'):
                            xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], 'cpu') else box.xyxy[0]
                            print(f"    Box {i}: {xyxy}")
                        if hasattr(box, 'conf'):
                            conf = box.conf[0].cpu().numpy() if hasattr(box.conf[0], 'cpu') else box.conf[0]
                            print(f"    Conf {i}: {conf}")
                        if hasattr(box, 'cls'):
                            cls = box.cls[0].cpu().numpy() if hasattr(box.cls[0], 'cpu') else box.cls[0]
                            print(f"    Class {i}: {cls} ({model.names.get(int(cls), 'Unknown')})")
                else:
                    print("  ⚠️ No boxes found")
            else:
                print("  ⚠️ No results")
        
        # Test 2: Image dengan multiple faces
        print("\n🔄 Test 2: Multiple faces image...")
        
        multi_face = np.random.randint(100, 150, (640, 640, 3), dtype=np.uint8)
        
        # Face 1 (kiri)
        cv2.circle(multi_face, (200, 200), 80, (255, 255, 255), -1)
        cv2.circle(multi_face, (180, 180), 15, (0, 0, 0), -1)
        cv2.circle(multi_face, (220, 180), 15, (0, 0, 0), -1)
        cv2.ellipse(multi_face, (200, 240), (25, 15), 0, 0, 180, (0, 0, 0), 3)
        
        # Face 2 (kanan)
        cv2.circle(multi_face, (440, 200), 80, (255, 255, 255), -1)
        cv2.circle(multi_face, (420, 180), 15, (0, 0, 0), -1)
        cv2.circle(multi_face, (460, 180), 15, (0, 0, 0), -1)
        cv2.ellipse(multi_face, (440, 240), (25, 15), 0, 0, 180, (0, 0, 0), 3)
        
        print(f"🖼️ Multi-face image shape: {multi_face.shape}")
        
        results_multi = model(multi_face, conf=0.01)
        
        if results_multi and len(results_multi) > 0:
            result_multi = results_multi[0]
            if hasattr(result_multi, 'boxes') and result_multi.boxes is not None:
                boxes_multi = result_multi.boxes
                print(f"  📦 Multi-face detections: {len(boxes_multi)}")
                
                for i, box in enumerate(boxes_multi):
                    if hasattr(box, 'xyxy'):
                        xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], 'cpu') else box.xyxy[0]
                        print(f"    Box {i}: {xyxy}")
                    if hasattr(box, 'conf'):
                        conf = box.conf[0].cpu().numpy() if hasattr(box.conf[0], 'cpu') else box.conf[0]
                        print(f"    Conf {i}: {conf}")
                    if hasattr(box, 'cls'):
                        cls = box.cls[0].cpu().numpy() if hasattr(box.cls[0], 'cpu') else box.cls[0]
                        print(f"    Class {i}: {cls} ({model.names.get(int(cls), 'Unknown')})")
            else:
                print("  ⚠️ No boxes found in multi-face")
        else:
            print("  ⚠️ No results for multi-face")
        
        # Test 3: Cek apakah model bisa detect dengan input yang berbeda
        print("\n🔄 Test 3: Different input formats...")
        
        # Test dengan image yang di-resize
        resized_face = cv2.resize(face_image, (416, 416))  # YOLO default size
        print(f"🖼️ Resized image shape: {resized_face.shape}")
        
        results_resized = model(resized_face, conf=0.01)
        
        if results_resized and len(results_resized) > 0:
            result_resized = results_resized[0]
            if hasattr(result_resized, 'boxes') and result_resized.boxes is not None:
                boxes_resized = result_resized.boxes
                print(f"  📦 Resized image detections: {len(boxes_resized)}")
            else:
                print("  ⚠️ No boxes in resized image")
        else:
            print("  ⚠️ No results for resized image")
        
        # Test 4: Cek model info lebih detail
        print("\n🔄 Test 4: Model information...")
        
        print(f"  📊 Model task: {model.task}")
        print(f"  📊 Model type: {type(model)}")
        print(f"  📊 Model device: {model.device}")
        
        # Cek apakah ada masalah dengan model weights
        if hasattr(model, 'model'):
            print(f"  📊 Inner model type: {type(model.model)}")
            if hasattr(model.model, 'names'):
                print(f"  📊 Inner model names: {model.model.names}")
        
    except Exception as e:
        print(f"❌ Error testing YOLO model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_with_realistic_images() 