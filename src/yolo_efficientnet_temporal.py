# File: src/yolo_efficientnet_temporal.py
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
import cv2
import numpy as np
from collections import deque
import face_recognition
import os
from ultralytics import YOLO
import time

class EmotionEfficientNetBiLSTM(nn.Module):
    """
    EfficientNet + BiLSTM khusus untuk klasifikasi emosi temporal
    Input: sequence of cropped faces dari YOLO
    Output: emotion classification
    """
    def __init__(self, num_emotions=8, lstm_hidden=256, num_layers=2, dropout=0.3):
        super(EmotionEfficientNetBiLSTM, self).__init__()
        
        # EfficientNet sebagai CNN feature extractor
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')
        
        # Hapus classifier layer, ambil feature dimension
        feature_dim = self.efficientnet._fc.in_features
        self.efficientnet._fc = nn.Identity()
        
        # Freeze early layers (optional, untuk faster training)
        for param in list(self.efficientnet.parameters())[:-10]:
            param.requires_grad = False
            
        # BiLSTM untuk temporal sequence modeling
        self.bilstm = nn.LSTM(
            input_size=feature_dim,  # 1280 untuk efficientnet-b0
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism (optional tapi sangat membantu)
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden * 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Final emotion classifier
        self.emotion_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(lstm_hidden),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, num_emotions)
        )
        
        self.emotion_labels = [
            'Anger', 'Contempt', 'Disgust', 'Fear', 
            'Happy', 'Neutral', 'Sad', 'Surprised'
        ]
        
    def forward(self, x):
        """
        x: (batch_size, sequence_length, 3, 224, 224) - sequence of face crops
        """
        batch_size, seq_len, c, h, w = x.shape
        
        # Reshape untuk process semua frames sekaligus
        x = x.view(batch_size * seq_len, c, h, w)
        
        # Extract CNN features menggunakan EfficientNet
        cnn_features = self.efficientnet(x)  # (batch_size * seq_len, feature_dim)
        
        # Reshape kembali untuk LSTM
        cnn_features = cnn_features.view(batch_size, seq_len, -1)
        
        # BiLSTM processing
        lstm_out, (hidden, cell) = self.bilstm(cnn_features)
        
        # Self-attention untuk focus pada frame penting
        attended_features, attention_weights = self.attention(
            lstm_out, lstm_out, lstm_out
        )
        
        # Ambil last timestep atau global average pooling
        # Option 1: Last timestep
        final_features = attended_features[:, -1, :]
        
        # Option 2: Global average pooling (uncomment jika mau coba)
        # final_features = torch.mean(attended_features, dim=1)
        
        # Emotion classification
        emotion_logits = self.emotion_classifier(final_features)
        
        return emotion_logits, attention_weights

class YOLOTemporalEmotionDetector:
    """
    Pipeline: YOLO Detection → EfficientNet BiLSTM Emotion Classification
    """
    def __init__(self, yolo_model_path, emotion_model_path=None, sequence_length=16):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load YOLO model untuk face detection
        self.yolo_model = YOLO(yolo_model_path)
        print(f"YOLO model loaded from {yolo_model_path}")
        
        # Initialize EfficientNet BiLSTM untuk emotion analysis
        self.emotion_model = EmotionEfficientNetBiLSTM()
        
        if emotion_model_path and os.path.exists(emotion_model_path):
            checkpoint = torch.load(emotion_model_path, map_location=self.device)
            self.emotion_model.load_state_dict(checkpoint)
            print(f"Emotion model loaded from {emotion_model_path}")
        else:
            print("No emotion model loaded, using random weights")
            
        self.emotion_model.to(self.device)
        self.emotion_model.eval()
        
        # Parameters
        self.sequence_length = sequence_length
        self.confidence_threshold = 0.5
        self.face_size = (224, 224)
        
        # Temporal buffers untuk setiap detected person
        self.person_buffers = {}  # {person_id: deque of face crops}
        self.person_locations = {}  # {person_id: last known location}
        self.person_emotions = {}  # {person_id: recent emotion history}
        self.frame_count = 0
        
        # Face recognition untuk tracking
        self.known_face_encodings, self.known_face_names = self.load_known_faces()
        
        # Image preprocessing untuk EfficientNet
        self.face_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.face_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
    def load_known_faces(self, known_faces_dir='known_faces'):
        """Load known faces untuk person tracking"""
        encodings, names = [], []
        
        if not os.path.exists(known_faces_dir):
            print(f"Known faces directory not found: {known_faces_dir}")
            return encodings, names
            
        for filename in os.listdir(known_faces_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(known_faces_dir, filename)
                name = os.path.splitext(filename)[0]
                
                try:
                    image = face_recognition.load_image_file(path)
                    face_encodings = face_recognition.face_encodings(image)
                    
                    if face_encodings:
                        encodings.append(face_encodings[0])
                        names.append(name)
                        print(f"Loaded face: {name}")
                        
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
                    
        return encodings, names
    
    def detect_faces_yolo(self, frame):
        """
        Detect faces menggunakan YOLO
        Returns: list of (x1, y1, x2, y2, confidence)
        """
        results = self.yolo_model(frame, verbose=False)
        detections = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    confidence = float(box.conf)
                    if confidence >= self.confidence_threshold:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        detections.append((x1, y1, x2, y2, confidence))
                        
        return detections
    
    def recognize_person(self, face_crop):
        """
        Recognize person from face crop
        Returns: person_name or "Unknown"
        """
        if len(self.known_face_encodings) == 0:
            return "Unknown"
            
        try:
            rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(rgb_face)
            
            if face_encodings:
                distances = face_recognition.face_distance(
                    self.known_face_encodings, face_encodings[0]
                )
                
                best_match_idx = np.argmin(distances)
                if distances[best_match_idx] < 0.6:  # threshold
                    return self.known_face_names[best_match_idx]
                    
        except Exception as e:
            print(f"Face recognition error: {e}")
            
        return "Unknown"
    
    def track_persons(self, current_detections, frame):
        """
        Track persons across frames menggunakan IoU dan face recognition
        Returns: list of (person_id, bbox, face_crop)
        """
        tracked_persons = []
        used_detections = set()
        
        # Match dengan existing persons berdasarkan IoU
        for person_id, last_bbox in self.person_locations.items():
            best_match = None
            best_iou = 0.3  # minimum IoU threshold
            
            for i, (x1, y1, x2, y2, conf) in enumerate(current_detections):
                if i in used_detections:
                    continue
                    
                # Calculate IoU dengan last known location
                iou = self.calculate_iou(last_bbox, (x1, y1, x2, y2))
                
                if iou > best_iou:
                    best_iou = iou
                    best_match = i
                    
            if best_match is not None:
                x1, y1, x2, y2, conf = current_detections[best_match]
                face_crop = frame[y1:y2, x1:x2]
                
                if face_crop.size > 0:
                    tracked_persons.append((person_id, (x1, y1, x2, y2), face_crop))
                    used_detections.add(best_match)
                    self.person_locations[person_id] = (x1, y1, x2, y2)
        
        # Handle new detections (yang belum di-match)
        for i, (x1, y1, x2, y2, conf) in enumerate(current_detections):
            if i not in used_detections:
                face_crop = frame[y1:y2, x1:x2]
                
                if face_crop.size > 0:
                    # Try face recognition untuk existing person
                    person_name = self.recognize_person(face_crop)
                    
                    if person_name in self.person_locations:
                        person_id = person_name
                    else:
                        # Create new person ID
                        person_id = f"{person_name}_{self.frame_count}_{i}"
                    
                    tracked_persons.append((person_id, (x1, y1, x2, y2), face_crop))
                    self.person_locations[person_id] = (x1, y1, x2, y2)
        
        return tracked_persons
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
            
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def preprocess_face_for_emotion(self, face_crop):
        """
        Preprocess face crop untuk EfficientNet
        Returns: tensor (3, 224, 224)
        """
        if face_crop.size == 0:
            return None
            
        try:
            # Convert BGR to RGB
            rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            
            # Apply transforms
            face_tensor = self.face_transform(rgb_face)
            
            return face_tensor
            
        except Exception as e:
            print(f"Face preprocessing error: {e}")
            return None
    
    def predict_temporal_emotion(self, person_id, face_tensor):
        """
        Predict emotion menggunakan temporal sequence
        Returns: (emotion, confidence, attention_weights)
        """
        if person_id not in self.person_buffers:
            self.person_buffers[person_id] = deque(maxlen=self.sequence_length)
            
        # Add current face tensor to buffer
        self.person_buffers[person_id].append(face_tensor)
        
        # Need minimum frames untuk prediction
        if len(self.person_buffers[person_id]) < 3:
            return "Neutral", 0.5, None
            
        # Prepare sequence tensor
        sequence = list(self.person_buffers[person_id])
        
        # Pad sequence jika kurang dari sequence_length
        while len(sequence) < self.sequence_length:
            sequence = [sequence[0]] + sequence  # Repeat first frame
            
        # Convert to tensor: (1, seq_len, 3, 224, 224)
        sequence_tensor = torch.stack(sequence).unsqueeze(0).to(self.device)
        
        try:
            with torch.no_grad():
                emotion_logits, attention_weights = self.emotion_model(sequence_tensor)
                probabilities = torch.softmax(emotion_logits, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
                
                emotion = self.emotion_model.emotion_labels[predicted_class.item()]
                conf_score = confidence.item()
                
                return emotion, conf_score, attention_weights
                
        except Exception as e:
            print(f"Emotion prediction error: {e}")
            return "Neutral", 0.5, None
    
    def smooth_emotion_predictions(self, person_id, emotion, confidence):
        """
        Temporal smoothing untuk stabilkan predictions
        """
        if person_id not in self.person_emotions:
            self.person_emotions[person_id] = deque(maxlen=5)  # 5 frame smoothing
            
        self.person_emotions[person_id].append((emotion, confidence))
        
        # Weighted voting berdasarkan confidence
        emotion_votes = {}
        total_weight = 0
        
        for emo, conf in self.person_emotions[person_id]:
            if emo not in emotion_votes:
                emotion_votes[emo] = 0
            emotion_votes[emo] += conf
            total_weight += conf
            
        # Normalize dan pilih emotion dengan vote tertinggi
        if total_weight > 0:
            best_emotion = max(emotion_votes, key=emotion_votes.get)
            best_confidence = emotion_votes[best_emotion] / total_weight
            return best_emotion, best_confidence
            
        return emotion, confidence
    
    def process_frame(self, frame):
        """
        Process single frame:
        1. YOLO face detection
        2. Person tracking
        3. EfficientNet BiLSTM emotion analysis
        """
        self.frame_count += 1
        results = []
        
        # Step 1: YOLO face detection
        face_detections = self.detect_faces_yolo(frame)
        
        if not face_detections:
            return results
            
        # Step 2: Track persons across frames
        tracked_persons = self.track_persons(face_detections, frame)
        
        # Step 3: Emotion analysis untuk setiap tracked person
        for person_id, bbox, face_crop in tracked_persons:
            # Preprocess face untuk emotion model
            face_tensor = self.preprocess_face_for_emotion(face_crop)
            
            if face_tensor is None:
                continue
                
            # Temporal emotion prediction
            emotion, confidence, attention_weights = self.predict_temporal_emotion(
                person_id, face_tensor
            )
            
            # Smooth predictions
            smoothed_emotion, smoothed_confidence = self.smooth_emotion_predictions(
                person_id, emotion, confidence
            )
            
            # Extract person name dari person_id
            person_name = person_id.split('_')[0] if '_' in person_id else person_id
            
            results.append({
                'person_id': person_id,
                'person_name': person_name,
                'emotion': smoothed_emotion,
                'confidence': smoothed_confidence,
                'bbox': bbox,
                'attention_weights': attention_weights
            })
            
        # Cleanup old persons yang tidak muncul lagi
        self.cleanup_old_persons(tracked_persons)
        
        return results
    
    def cleanup_old_persons(self, current_persons, max_absent_frames=30):
        """Clean up persons yang sudah lama tidak muncul"""
        current_ids = {person_id for person_id, _, _ in current_persons}
        
        ids_to_remove = []
        for person_id in self.person_buffers:
            if person_id not in current_ids:
                # Mark untuk removal setelah beberapa frame
                ids_to_remove.append(person_id)
        
        # Remove setelah threshold tertentu (implementasi sederhana)
        if len(ids_to_remove) > 10:  # Keep max 10 absent persons
            for person_id in ids_to_remove[:5]:  # Remove oldest 5
                if person_id in self.person_buffers:
                    del self.person_buffers[person_id]
                if person_id in self.person_locations:
                    del self.person_locations[person_id]
                if person_id in self.person_emotions:
                    del self.person_emotions[person_id]
    
    def draw_results(self, frame, results):
        """Draw detection dan emotion results pada frame"""
        emotion_colors = {
            'Anger': (0, 0, 255),      # Red
            'Contempt': (0, 128, 255), # Orange  
            'Disgust': (0, 255, 0),    # Green
            'Fear': (128, 0, 128),     # Purple
            'Happy': (0, 255, 255),    # Yellow
            'Neutral': (255, 255, 255),# White
            'Sad': (255, 0, 0),        # Blue
            'Surprised': (255, 165, 0) # Cyan
        }
        
        for result in results:
            person_name = result['person_name']
            emotion = result['emotion']
            confidence = result['confidence']
            x1, y1, x2, y2 = result['bbox']
            
            # Color berdasarkan emotion
            color = emotion_colors.get(emotion, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            conf_percent = confidence * 100
            label = f"{person_name}: {emotion} ({conf_percent:.1f}%)"
            
            # Buffer length info (untuk debugging)
            person_id = result['person_id']
            buffer_len = len(self.person_buffers.get(person_id, []))
            debug_label = f"Frames: {buffer_len}/{self.sequence_length}"
            
            # Draw main label
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            cv2.rectangle(frame, (x1, y1 - text_height - 15), 
                         (x1 + text_width + 10, y1), (0, 0, 0), -1)
            
            cv2.putText(frame, label, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw debug info
            cv2.putText(frame, debug_label, (x1, y2 + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                       
        # Draw frame info
        info_text = f"Frame: {self.frame_count} | Active persons: {len(results)}"
        cv2.putText(frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame

# Demo usage
def main():
    """Demo penggunaan YOLO → EfficientNet BiLSTM pipeline"""
    
    # Initialize detector
    detector = YOLOTemporalEmotionDetector(
        yolo_model_path='yolov8n-face.pt',  # Ganti dengan path YOLO model Anda
        emotion_model_path=None,  # Optional: path ke trained emotion model
        sequence_length=16  # 16 frames untuk temporal analysis
    )
    
    # Open video source
    cap = cv2.VideoCapture(0)  # Webcam, atau ganti dengan video file
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("Starting real-time emotion detection...")
    print("Press 'q' to quit")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            start_time = time.time()
            
            # Process frame dengan pipeline YOLO → EfficientNet BiLSTM
            results = detector.process_frame(frame)
            
            # Draw results
            output_frame = detector.draw_results(frame, results)
            
            # Calculate FPS
            fps = 1.0 / (time.time() - start_time)
            cv2.putText(output_frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Display
            cv2.imshow('YOLO → EfficientNet BiLSTM Emotion Detection', output_frame)
            
            # Print results (optional)
            if results:
                for result in results:
                    print(f"{result['person_name']}: {result['emotion']} "
                          f"({result['confidence']:.2f})")
            
            # Exit condition
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nStopping...")
        
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Demo finished.")

if __name__ == "__main__":
    main()