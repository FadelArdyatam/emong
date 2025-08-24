"""
Flask Application dengan Hybrid Emotion Detector
Mengkombinasikan rule-based dan trained model untuk akurasi maksimal
"""

from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import os
import time
from datetime import datetime
import json
import base64
from io import BytesIO
from PIL import Image

# Import hybrid detector
from src.hybrid_emotion_detector import HybridEmotionDetector

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hybrid_emotion_detection_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize hybrid detector
print("🚀 Initializing Hybrid Emotion Detection System...")
hybrid_detector = HybridEmotionDetector(
    trained_model_path=None,  # Will be set when model is ready
    use_mediapipe=True,
    ensemble_weight=0.7
)
print("✅ Hybrid detector initialized!")

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/hybrid')
def hybrid_dashboard():
    """Hybrid detection dashboard"""
    return render_template('hybrid_dashboard.html')

@app.route('/capture')
def capture():
    """Photo capture page"""
    return render_template('capture.html')

@app.route('/upload')
def upload():
    """Image upload page"""
    return render_template('upload.html')

@app.route('/debug')
def debug():
    """Debug page for testing"""
    return render_template('debug_realtime.html')

@app.route('/api/hybrid-stats')
def get_hybrid_stats():
    """Get hybrid detection statistics"""
    try:
        stats = hybrid_detector.get_hybrid_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/trained-model-info')
def get_trained_model_info():
    """Get trained model information"""
    try:
        model_info = hybrid_detector.trained_detector.get_model_info()
        return jsonify(model_info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/accuracy-stats')
def get_accuracy_stats():
    """Get accuracy statistics"""
    try:
        accuracy_stats = hybrid_detector.trained_detector.get_accuracy_stats()
        return jsonify(accuracy_stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/set-model-path', methods=['POST'])
def set_model_path():
    """Set trained model path"""
    try:
        data = request.get_json()
        model_path = data.get('model_path')
        
        if model_path and os.path.exists(model_path):
            # Reinitialize hybrid detector with new model
            global hybrid_detector
            hybrid_detector = HybridEmotionDetector(
                trained_model_path=model_path,
                use_mediapipe=True,
                ensemble_weight=0.7
            )
            
            return jsonify({
                'success': True,
                'message': f'Model loaded from: {model_path}',
                'model_info': hybrid_detector.trained_detector.get_model_info()
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Model path not found'
            }), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/process-capture', methods=['POST'])
def process_capture():
    """Process captured photo dengan hybrid approach"""
    try:
        # Get image data
        image_data = request.json.get('image')
        if not image_data:
            return jsonify({'error': 'No image data'}), 400
        
        # Decode base64 image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Convert to OpenCV format
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Process with hybrid detector
        start_time = time.time()
        results = hybrid_detector.process_image_hybrid(image)
        processing_time = time.time() - start_time
        
        # Draw bounding boxes on image
        processed_image = draw_bounding_boxes_on_image(image, results['detections'])
        
        # Save processed image
        timestamp = int(time.time())
        filename = f"hybrid_capture_{timestamp}.jpg"
        filepath = os.path.join('uploads', filename)
        
        cv2.imwrite(filepath, processed_image)
        
        # Create response
        response = {
            'success': True,
            'detections': results['detections'],
            'total_faces': results['total_faces'],
            'processing_time': processing_time,
            'detection_method': 'hybrid',
            'trained_model_loaded': results['trained_model_loaded'],
            'ensemble_weight': results['ensemble_weight'],
            'processed_image': f"/uploads/{filename}",
            'hybrid_stats': hybrid_detector.get_hybrid_stats()
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error processing capture: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/process-upload', methods=['POST'])
def process_upload():
    """Process uploaded image dengan hybrid approach"""
    try:
        # Get uploaded file
        if 'image' not in request.files:
            return jsonify({'error': 'No image file'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Process with hybrid detector
        start_time = time.time()
        results = hybrid_detector.process_image_hybrid(image)
        processing_time = time.time() - start_time
        
        # Save original and processed images
        timestamp = int(time.time())
        original_filename = f"upload_{timestamp}_{file.filename}"
        processed_filename = f"result_hybrid_{timestamp}_{file.filename}"
        
        original_path = os.path.join('uploads', original_filename)
        processed_path = os.path.join('uploads', processed_filename)
        
        # Save original
        with open(original_path, 'wb') as f:
            f.write(image_bytes)
        
        # Draw bounding boxes and save processed
        processed_image = draw_bounding_boxes_on_image(image, results['detections'])
        cv2.imwrite(processed_path, processed_image)
        
        # Create response
        response = {
            'success': True,
            'detections': results['detections'],
            'total_faces': results['total_faces'],
            'processing_time': processing_time,
            'detection_method': 'hybrid',
            'trained_model_loaded': results['trained_model_loaded'],
            'ensemble_weight': results['ensemble_weight'],
            'original_image': f"/uploads/{original_filename}",
            'processed_image': f"/uploads/{processed_filename}",
            'hybrid_stats': hybrid_detector.get_hybrid_stats()
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error processing upload: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/save-hybrid-results')
def save_hybrid_results():
    """Save hybrid detection results"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hybrid_results_{timestamp}.json"
        filepath = os.path.join('uploads', filename)
        
        hybrid_detector.save_hybrid_results(filepath)
        
        return jsonify({
            'success': True,
            'message': f'Results saved to: {filename}',
            'filepath': filepath
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/reset-hybrid-stats')
def reset_hybrid_stats():
    """Reset hybrid detection statistics"""
    try:
        hybrid_detector.reset_stats()
        return jsonify({
            'success': True,
            'message': 'Hybrid statistics reset'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Socket.IO events
@socketio.on('connect')
def handle_connect():
    print('Client connected to hybrid system')
    emit('connection_status', {'status': 'connected', 'method': 'hybrid'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected from hybrid system')

@socketio.on('frame_data')
def handle_frame_processing(data):
    """Handle real-time frame processing"""
    try:
        # Decode image data
        image_data = data.get('image')
        if not image_data:
            emit('frame_result', {'error': 'No image data'})
            return
        
        # Remove data URL prefix
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            emit('frame_result', {'error': 'Invalid image data'})
            return
        
        # Process with hybrid detector
        results = hybrid_detector.process_image_hybrid(image)
        
        # Emit results
        emit('frame_result', results)
        
    except Exception as e:
        print(f"Frame processing error: {e}")
        emit('frame_result', {'error': str(e)})

# Helper functions
def draw_bounding_boxes_on_image(image: np.ndarray, detections: list) -> np.ndarray:
    """Draw bounding boxes and labels on image"""
    result_image = image.copy()
    
    for detection in detections:
        bbox = detection['bbox']
        emotion = detection['emotion']
        confidence = detection['emotion_confidence']
        ensemble_method = detection.get('ensemble_method', 'unknown')
        
        # Get emotion color
        color = get_emotion_color(emotion)
        
        # Draw bounding box
        x, y, w, h = bbox
        cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
        
        # Create label
        label = f"{emotion} ({confidence:.2f})"
        if ensemble_method != 'unknown':
            label += f" [{ensemble_method}]"
        
        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(result_image, (x, y - label_size[1] - 10), 
                     (x + label_size[0], y), color, -1)
        
        # Draw label text
        cv2.putText(result_image, label, (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw emotion emoji
        emoji = get_emotion_emoji(emotion)
        cv2.putText(result_image, emoji, (x + w - 30, y + h + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    return result_image

def get_emotion_color(emotion: str) -> tuple:
    """Get color for emotion"""
    colors = {
        'Happy': (0, 255, 0),      # Green
        'Sad': (255, 0, 0),        # Red
        'Angry': (0, 0, 255),      # Blue
        'Surprised': (255, 255, 0), # Cyan
        'Fear': (128, 0, 128),     # Purple
        'Disgust': (0, 128, 128),  # Teal
        'Neutral': (128, 128, 128) # Gray
    }
    return colors.get(emotion, (255, 255, 255))

def get_emotion_emoji(emotion: str) -> str:
    """Get emoji for emotion"""
    emojis = {
        'Happy': '😊',
        'Sad': '😢',
        'Angry': '😠',
        'Surprised': '😲',
        'Fear': '😨',
        'Disgust': '🤢',
        'Neutral': '😐'
    }
    return emojis.get(emotion, '❓')

if __name__ == '__main__':
    print("🚀 Starting Hybrid Emotion Detection Flask App...")
    print("📊 Available routes:")
    print("   - / (Main page)")
    print("   - /hybrid (Hybrid dashboard)")
    print("   - /capture (Photo capture)")
    print("   - /upload (Image upload)")
    print("   - /debug (Debug page)")
    print("   - /api/hybrid-stats (Hybrid statistics)")
    print("   - /api/trained-model-info (Model info)")
    print("   - /api/accuracy-stats (Accuracy stats)")
    print("   - /api/set-model-path (Set model path)")
    print("   - /api/process-capture (Process capture)")
    print("   - /api/process-upload (Process upload)")
    print("   - /api/save-hybrid-results (Save results)")
    print("   - /api/reset-hybrid-stats (Reset stats)")
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000) 