#!/usr/bin/env python3
"""
Debug script untuk melihat output dari model YOLO
"""

import cv2
import numpy as np
from ultralytics import YOLO
import torch

def test_yolo_model():
    """Test YOLO model dengan image sederhana"""
    
    # Path ke model
    model_path = "models/yolo_emotion_detection_v8s.pt"
    
    print(f"🔄 Testing YOLO model: {model_path}")
    
    try:
        # Load model
        model = YOLO(model_path)
        print(f"✅ Model loaded successfully!")
        print(f"📊 Classes: {model.names}")
        print(f"🔢 Number of classes: {len(model.names)}")
        
        # Buat test image sederhana (hitam dengan kotak putih)
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Tambah kotak putih di tengah (simulasi wajah)
        cv2.rectangle(test_image, (200, 150), (440, 330), (255, 255, 255), -1)
        
        print(f"🖼️ Test image shape: {test_image.shape}")
        
        # Inference
        print("🔄 Running inference...")
        results = model(test_image, conf=0.3)
        
        print(f"📊 Results type: {type(results)}")
        print(f"📊 Results length: {len(results)}")
        
        if results and len(results) > 0:
            result = results[0]
            print(f"📊 First result type: {type(result)}")
            
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                print(f"📦 Boxes count: {len(boxes)}")
                print(f"📦 Boxes type: {type(boxes)}")
                
                if len(boxes) > 0:
                    print(f"📦 First box data: {boxes.data[0] if hasattr(boxes, 'data') else 'No data'}")
                    print(f"📦 First box xyxy: {boxes.xyxy[0] if hasattr(boxes, 'xyxy') else 'No xyxy'}")
                    print(f"📦 First box conf: {boxes.conf[0] if hasattr(boxes, 'conf') else 'No conf'}")
                    print(f"📦 First box cls: {boxes.cls[0] if hasattr(boxes, 'cls') else 'No cls'}")
                else:
                    print("⚠️ No detections found")
            else:
                print("⚠️ No boxes attribute found")
        else:
            print("⚠️ No results returned")
            
        # Test dengan image yang lebih realistis
        print("\n🔄 Testing with more realistic image...")
        
        # Buat image dengan gradien (lebih realistis)
        realistic_image = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
        
        # Tambah wajah sederhana
        cv2.circle(realistic_image, (320, 240), 80, (255, 255, 255), -1)  # Kepala
        cv2.circle(realistic_image, (300, 220), 10, (0, 0, 0), -1)       # Mata kiri
        cv2.circle(realistic_image, (340, 220), 10, (0, 0, 0), -1)       # Mata kanan
        cv2.ellipse(realistic_image, (320, 260), (30, 15), 0, 0, 180, (0, 0, 0), -1)  # Mulut
        
        results2 = model(realistic_image, conf=0.1)  # Lower confidence threshold
        
        print(f"📊 Realistic image results: {len(results2) if results2 else 0}")
        
        if results2 and len(results2) > 0:
            result2 = results2[0]
            if hasattr(result2, 'boxes') and result2.boxes is not None:
                boxes2 = result2.boxes
                print(f"📦 Realistic detections: {len(boxes2)}")
                
                for i, box in enumerate(boxes2):
                    if hasattr(box, 'xyxy'):
                        xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], 'cpu') else box.xyxy[0]
                        print(f"  Box {i}: {xyxy}")
                    if hasattr(box, 'conf'):
                        conf = box.conf[0].cpu().numpy() if hasattr(box.conf[0], 'cpu') else box.conf[0]
                        print(f"  Conf {i}: {conf}")
                    if hasattr(box, 'cls'):
                        cls = box.cls[0].cpu().numpy() if hasattr(box.cls[0], 'cpu') else box.cls[0]
                        print(f"  Class {i}: {cls} ({model.names.get(int(cls), 'Unknown')})")
        
        # Test dengan confidence yang sangat rendah
        print("\n🔄 Testing with very low confidence...")
        results3 = model(realistic_image, conf=0.01)
        
        if results3 and len(results3) > 0:
            result3 = results3[0]
            if hasattr(result3, 'boxes') and result3.boxes is not None:
                boxes3 = result3.boxes
                print(f"📦 Very low conf detections: {len(boxes3)}")
                
                for i, box in enumerate(boxes3):
                    if hasattr(box, 'xyxy'):
                        xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], 'cpu') else box.xyxy[0]
                        print(f"  Box {i}: {xyxy}")
                    if hasattr(box, 'conf'):
                        conf = box.conf[0].cpu().numpy() if hasattr(box.conf[0], 'cpu') else box.conf[0]
                        print(f"  Conf {i}: {conf}")
        
    except Exception as e:
        print(f"❌ Error testing YOLO model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_yolo_model() 