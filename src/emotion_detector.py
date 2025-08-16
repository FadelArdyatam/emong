# File: src/emotion_detector.py
import cv2
import numpy as np

def detect_emotion(model, image, confidence_threshold=0.05):
    print("Running emotion detection...")
    try:
        results = model(image)
        print(f"YOLO results: {results}")
        print(f"Model classes: {model.names}")

        detections = []
        if not results:
            print("No results returned from YOLO model.")
            return []
            
        for result in results:
            boxes = result.boxes if hasattr(result, 'boxes') else []
            print(f"Number of boxes detected: {len(boxes)}")
            if not boxes:
                print("No boxes detected in YOLO results.")
                continue
                
            for box in boxes:
                conf = float(box.conf)
                print(f"Box confidence: {conf}")
                if conf < confidence_threshold:
                    print(f"Confidence {conf} below threshold {confidence_threshold}, skipping.")
                    continue
                    
                cls = int(box.cls)
                label = model.names[cls]
                print(f"Detected: {label}, Confidence: {conf}")
                
                valid_emotions = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprised']
                if label not in valid_emotions:
                    label = 'Unknown'
                    print(f"Label {label} not in valid emotions, set to Unknown.")
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                print(f"Bounding box coordinates: ({x1}, {y1}, {x2}, {y2})")
                detections.append([label, conf, (x1, y1, x2, y2)])
        
        if not detections:
            print("No faces detected with confidence above threshold.")
            return []
        
        print(f"Final detections: {detections}")
        return detections
    except Exception as e:
        print(f"Error in detect_emotion: {e}")
        return []

def draw_bounding_box(image, detections, emotion_colors_bgr):
    for emotion, confidence, (x1, y1, x2, y2) in detections:
        color = emotion_colors_bgr.get(emotion, (255, 255, 255))
        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        
        # Chuyển confidence sang %
        confidence_percent = confidence * 100
        label = f"{emotion}: {confidence_percent:.0f}%"
        
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        
        text_x = x1
        text_y = y1 - 10 if y1 - text_height - 10 > 0 else y2 + text_height + 10
        box_y = y1 - text_height - 15 if y1 - text_height - 10 > 0 else y2 + 5
        
        cv2.rectangle(image, (text_x, box_y), (text_x + text_width + 10, box_y + text_height + 10), (0, 0, 0), -1)
        
        cv2.putText(image, label, (text_x + 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return image