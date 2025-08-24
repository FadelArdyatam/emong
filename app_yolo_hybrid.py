#!/usr/bin/env python3
"""
YOLO Hybrid Emotion Detection Flask Application
Menggabungkan YOLO dan rule-based detection untuk emotion recognition
"""

import cv2
import numpy as np
import json
import os
import base64
from datetime import datetime
import time
from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO, emit
import threading
import queue

# Import local modules
from src.yolo_hybrid_detector import YOLOHybridDetector

app = Flask(__name__)
app.config['SECRET_KEY'] = 'yolo_hybrid_secret_key_2024'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
hybrid_detector = None
detection_active = False
frame_queue = queue.Queue(maxsize=10)
stats_lock = threading.Lock()

def initialize_detector():
    """Initialize YOLO hybrid detector"""
    global hybrid_detector
    try:
        print("🚀 Initializing YOLO Hybrid Emotion Detection System...")
        hybrid_detector = YOLOHybridDetector(
            yolo_model_path="models/yolo_emotion_detection_v8s.pt",
            yolo_weight=0.8,
            rule_weight=0.2,
            confidence_threshold=0.3
        )
        print("✅ YOLO Hybrid Detector initialized successfully!")
        return True
    except Exception as e:
        print(f"❌ Error initializing detector: {e}")
        return False

def convert_numpy_types(obj):
    """Convert NumPy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def draw_bounding_boxes_on_image(image, detections):
    """Draw bounding boxes dan emotion labels pada image"""
    try:
        # Convert to BGR for OpenCV
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume RGB, convert to BGR
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image.copy()
        
        for detection in detections:
            bbox = detection.get('bbox', [])
            if len(bbox) == 4:
                x, y, w, h = bbox
                x, y, w, h = int(x), int(y), int(w), int(h)
                
                # Draw bounding box
                color = (0, 255, 0)  # Green
                thickness = 2
                cv2.rectangle(image_bgr, (x, y), (x + w, y + h), color, thickness)
                
                # Get emotion info
                emotion = detection.get('emotion', 'Unknown')
                confidence = detection.get('emotion_confidence', 0.0)
                source = detection.get('source', 'Unknown')
                
                # Create label
                label = f"{emotion} ({confidence:.2f}) [{source}]"
                
                # Draw label background
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(image_bgr, (x, y - label_size[1] - 10), 
                             (x + label_size[0], y), color, -1)
                
                # Draw label text
                cv2.putText(image_bgr, label, (x, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Convert back to RGB
        result_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return result_image
        
    except Exception as e:
        print(f"❌ Error drawing bounding boxes: {e}")
        return image

def detection_worker():
    """Background worker untuk processing frames"""
    global detection_active, hybrid_detector
    
    while detection_active:
        try:
            if not frame_queue.empty():
                frame_data = frame_queue.get(timeout=1)
                
                # Process frame
                if hybrid_detector:
                    results = hybrid_detector.detect_emotions(frame_data)
                    
                    # Convert NumPy types
                    results = convert_numpy_types(results)
                    
                    # Emit results via Socket.IO
                    socketio.emit('detection_results', results)
                    
                    # Add small delay to prevent overwhelming
                    time.sleep(0.1)
                else:
                    time.sleep(0.1)
            else:
                time.sleep(0.1)
                
        except Exception as e:
            print(f"❌ Error in detection worker: {e}")
            time.sleep(0.1)

@app.route('/')
def index():
    """Main dashboard"""
    return render_template('yolo_hybrid_dashboard.html')

@app.route('/api/hybrid/stats')
def get_hybrid_stats():
    """Get hybrid detection statistics"""
    if hybrid_detector is None:
        return jsonify({'error': 'Detector not initialized'})
    
    try:
        stats = hybrid_detector.get_detection_stats()
        return jsonify(convert_numpy_types(stats))
    except Exception as e:
        return jsonify({'error': f'Error getting stats: {str(e)}'})

@app.route('/api/yolo/model-info')
def get_yolo_model_info():
    """Get YOLO model information"""
    if hybrid_detector is None:
        return jsonify({'error': 'Detector not initialized'})
    
    try:
        model_info = hybrid_detector.get_yolo_model_info()
        return jsonify(model_info)
    except Exception as e:
        return jsonify({'error': f'Error getting model info: {str(e)}'})

@app.route('/api/hybrid/weights', methods=['POST'])
def update_weights():
    """Update detection weights"""
    if hybrid_detector is None:
        return jsonify({'error': 'Detector not initialized'})
    
    try:
        data = request.get_json()
        yolo_weight = float(data.get('yolo_weight', 0.8))
        rule_weight = float(data.get('rule_weight', 0.2))
        
        success = hybrid_detector.update_weights(yolo_weight, rule_weight)
        
        if success:
            return jsonify({'success': True, 'message': 'Weights updated'})
        else:
            return jsonify({'success': False, 'message': 'Invalid weights'})
            
    except Exception as e:
        return jsonify({'error': f'Error updating weights: {str(e)}'})

@app.route('/api/hybrid/confidence', methods=['POST'])
def update_confidence():
    """Update confidence threshold"""
    if hybrid_detector is None:
        return jsonify({'error': 'Detector not initialized'})
    
    try:
        data = request.get_json()
        threshold = float(data.get('confidence_threshold', 0.3))
        
        success = hybrid_detector.update_confidence_threshold(threshold)
        
        if success:
            return jsonify({'success': True, 'message': 'Confidence threshold updated'})
        else:
            return jsonify({'success': False, 'message': 'Invalid threshold'})
            
    except Exception as e:
        return jsonify({'error': f'Error updating confidence: {str(e)}'})

@app.route('/api/hybrid/reset', methods=['POST'])
def reset_stats():
    """Reset detection statistics"""
    if hybrid_detector is None:
        return jsonify({'error': 'Detector not initialized'})
    
    try:
        hybrid_detector.reset_stats()
        return jsonify({'success': True, 'message': 'Statistics reset'})
    except Exception as e:
        return jsonify({'error': f'Error resetting stats: {str(e)}'})

@app.route('/api/hybrid/temporal')
def get_temporal_data():
    """Get temporal data untuk charts"""
    if hybrid_detector is None:
        return jsonify({'error': 'Detector not initialized'})
    
    try:
        time_window = request.args.get('time_window', 300, type=int)
        temporal_data = hybrid_detector.get_temporal_data(time_window)
        return jsonify(convert_numpy_types(temporal_data))
    except Exception as e:
        return jsonify({'error': f'Error getting temporal data: {str(e)}'})

@app.route('/api/hybrid/start', methods=['POST'])
def start_detection():
    """Start real-time detection"""
    global detection_active
    
    try:
        detection_active = True
        
        # Start detection worker thread
        worker_thread = threading.Thread(target=detection_worker, daemon=True)
        worker_thread.start()
        
        return jsonify({'success': True, 'message': 'Detection started'})
    except Exception as e:
        return jsonify({'error': f'Error starting detection: {str(e)}'})

@app.route('/api/hybrid/stop', methods=['POST'])
def stop_detection():
    """Stop real-time detection"""
    global detection_active
    
    try:
        detection_active = False
        return jsonify({'success': True, 'message': 'Detection stopped'})
    except Exception as e:
        return jsonify({'error': f'Error stopping detection: {str(e)}'})

@app.route('/api/hybrid/capture', methods=['POST'])
def capture_image():
    """Capture dan process single image"""
    if hybrid_detector is None:
        return jsonify({'error': 'Detector not initialized'})
    
    try:
        # Get image data from request
        data = request.get_json()
        image_data = data.get('image_data', '')
        
        if not image_data:
            return jsonify({'error': 'No image data provided'})
        
        # Remove data URL prefix
        if image_data.startswith('data:image/'):
            image_data = image_data.split(',')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image data'})
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image
        results = hybrid_detector.detect_emotions(image_rgb)
        
        # Draw bounding boxes
        if results.get('faces_detected', 0) > 0:
            annotated_image = draw_bounding_boxes_on_image(image_rgb, results.get('emotions', []))
            
            # Convert back to base64
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            annotated_image_b64 = base64.b64encode(buffer).decode('utf-8')
            
            results['annotated_image'] = f"data:image/jpeg;base64,{annotated_image_b64}"
        
        # Convert NumPy types
        results = convert_numpy_types(results)
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'})

@app.route('/api/hybrid/upload', methods=['POST'])
def upload_image():
    """Upload dan process image file"""
    if hybrid_detector is None:
        return jsonify({'error': 'Detector not initialized'})
    
    try:
        # Check if file was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image file uploaded'})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        # Read image
        image_bytes = file.read()
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image file'})
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image
        results = hybrid_detector.detect_emotions(image_rgb)
        
        # Draw bounding boxes
        if results.get('faces_detected', 0) > 0:
            annotated_image = draw_bounding_boxes_on_image(image_rgb, results.get('emotions', []))
            
            # Convert back to base64
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            annotated_image_b64 = base64.b64encode(buffer).decode('utf-8')
            
            results['annotated_image'] = f"data:image/jpeg;base64,{annotated_image_b64}"
        
        # Convert NumPy types
        results = convert_numpy_types(results)
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': f'Error processing uploaded image: {str(e)}'})

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print(f"🔌 Client connected: {request.sid}")
    emit('connection_status', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print(f"🔌 Client disconnected: {request.sid}")

@socketio.on('frame_data')
def handle_frame_data(data):
    """Handle incoming frame data from client"""
    try:
        # Decode base64 image
        image_data = data.get('image_data', '')
        if image_data.startswith('data:image/'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is not None:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Add to processing queue
            if not frame_queue.full():
                frame_queue.put(image_rgb)
            
    except Exception as e:
        print(f"❌ Error handling frame data: {e}")

@socketio.on('request_stats')
def handle_stats_request():
    """Handle stats request from client"""
    if hybrid_detector is None:
        emit('stats_update', {'error': 'Detector not initialized'})
        return
    
    try:
        stats = hybrid_detector.get_detection_stats()
        emit('stats_update', convert_numpy_types(stats))
    except Exception as e:
        emit('stats_update', {'error': f'Error getting stats: {str(e)}'})

if __name__ == '__main__':
    # Initialize detector
    if initialize_detector():
        print("🚀 Starting YOLO Hybrid Flask application...")
        socketio.run(app, host='0.0.0.0', port=5000, debug=True)
    else:
        print("❌ Failed to initialize detector. Exiting.") 