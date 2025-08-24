# File: app_simple.py
"""
Aplikasi Flask dengan Simple Emotion Detector
Menggunakan OpenCV + Haar Cascade sebagai fallback
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_socketio import SocketIO
import cv2
import numpy as np
import os
import time
import base64
from src.simple_emotion_detector import FlaskSimpleIntegration

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
socketio = SocketIO(app, async_mode="threading")

# Membuat folder uploads jika belum ada
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize Simple Emotion Detector
print("🚀 Initializing Simple Emotion Detector...")
emotion_detector = FlaskSimpleIntegration()  # No cascade path needed
print("✅ Simple Emotion Detector ready!")

# Emotion emojis dan colors
EMOTION_EMOJIS = {
    "Happy": "😊", "Neutral": "😐", "Sad": "😢", 
    "Angry": "😠", "Surprised": "😲", "Unknown": "❓"
}

EMOTION_COLORS = {
    "Happy": "rgb(0, 255, 0)", "Neutral": "rgb(0, 0, 255)", 
    "Sad": "rgb(128, 128, 128)", "Angry": "rgb(255, 0, 0)", 
    "Surprised": "rgb(255, 0, 255)", "Unknown": "rgb(255, 255, 255)"
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture')
def capture():
    return render_template('capture.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/record')
def record():
    return render_template('record.html')

@app.route('/realtime')
def realtime():
    return render_template('realtime_advanced.html')  # Changed to advanced template

@app.route('/debug')
def debug():
    return render_template('debug_realtime.html')

@app.route('/capture', methods=['POST'])
def process_capture():
    """
    Process captured photo menggunakan Simple Emotion Detector
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        confidence = float(request.form.get('confidence', 0.5))
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        # Read image data
        image_data = file.read()
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image file'})
        
        # Process image dengan simple detector
        results = emotion_detector.detector.process_image(image)
        
        if 'error' in results:
            return jsonify(results)
        
        # Draw bounding boxes on image
        processed_image = draw_bounding_boxes_on_image(image, results.get('detections', []))
        
        # Save processed image
        timestamp = int(time.time())
        filename = f"capture_{timestamp}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        cv2.imwrite(filepath, processed_image)
        
        # Prepare response
        response = {
            'success': True,
            'image': f'/uploads/{filename}',
            'results': results,
            'processing_time': results.get('processing_time', 0),
            'total_faces': results.get('total_faces', 0),
            'detections': results.get('detections', [])
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'})

@app.route('/upload', methods=['POST'])
def process_upload():
    """
    Process uploaded image menggunakan Simple Emotion Detector
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        confidence = float(request.form.get('confidence', 0.5))
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        # Read image data
        image_data = file.read()
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image file'})
        
        # Process image dengan simple detector
        results = emotion_detector.detector.process_image(image)
        
        if 'error' in results:
            return jsonify(results)
        
        # Save original image
        timestamp = int(time.time())
        original_filename = f"upload_{timestamp}_{file.filename}"
        original_filepath = os.path.join(UPLOAD_FOLDER, original_filename)
        cv2.imwrite(original_filepath, image)
        
        # Draw bounding boxes on image
        processed_image = draw_bounding_boxes_on_image(image, results.get('detections', []))
        
        # Save processed image
        processed_filename = f"result_{timestamp}_{file.filename}"
        processed_filepath = os.path.join(UPLOAD_FOLDER, processed_filename)
        cv2.imwrite(processed_filepath, processed_image)
        
        # Prepare response
        response = {
            'success': True,
            'original_image': f'/uploads/{original_filename}',
            'processed_image': f'/uploads/{processed_filename}',
            'results': results,
            'processing_time': results.get('processing_time', 0),
            'total_faces': results.get('total_faces', 0),
            'detections': results.get('detections', [])
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'})

@app.route('/realtime', methods=['POST'])
def process_realtime():
    """
    Process realtime frame menggunakan Simple Emotion Detector
    """
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data received'})
        
        # Decode base64 image
        image_data = base64.b64decode(data['image'].split(',')[1])
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image data'})
        
        # Process frame dengan simple detector
        results = emotion_detector.detector.process_image(image)
        
        # Prepare response
        response = {
            'success': True,
            'results': results,
            'processing_time': results.get('processing_time', 0),
            'total_faces': results.get('total_faces', 0),
            'detections': results.get('detections', [])
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'})

@app.route('/emotion-summary')
def emotion_summary():
    """
    Get emotion summary dari Simple Emotion Detector
    """
    try:
        summary = emotion_detector.get_emotion_summary()
        return jsonify(summary)
    except Exception as e:
        return jsonify({'error': f'Error getting summary: {str(e)}'})

@app.route('/api/storage-info')
def api_storage_info():
    """
    Get storage info dari Simple Emotion Detector
    """
    try:
        storage_info = emotion_detector.get_data_storage_info()
        return jsonify(storage_info)
    except Exception as e:
        return jsonify({'error': f'Error getting storage info: {str(e)}'})

@app.route('/api/temporal-data')
def api_temporal_data():
    """
    Get temporal data untuk chart
    """
    try:
        face_id = request.args.get('face_id', type=int)
        time_window = request.args.get('time_window', 300, type=int)
        
        temporal_data = emotion_detector.get_temporal_data(face_id, time_window)
        return jsonify(temporal_data)
    except Exception as e:
        return jsonify({'error': f'Error getting temporal data: {str(e)}'})

@app.route('/reset-tracking', methods=['POST'])
def reset_tracking():
    """
    Reset tracking data
    """
    try:
        emotion_detector.reset_tracking()
        return jsonify({'success': True, 'message': 'Tracking data reset successfully'})
    except Exception as e:
        return jsonify({'error': f'Error resetting tracking: {str(e)}'})

@app.route('/api/emotion-distribution')
def api_emotion_distribution():
    """
    Get emotion distribution statistics untuk debugging bias
    """
    try:
        distribution_stats = emotion_detector.detector.get_emotion_distribution_stats()
        return jsonify(distribution_stats)
    except Exception as e:
        return jsonify({'error': f'Error getting emotion distribution: {str(e)}'})

@app.route('/api/reset-emotion-bias', methods=['POST'])
def api_reset_emotion_bias():
    """
    Reset emotion bias
    """
    try:
        emotion_detector.detector.reset_emotion_bias()
        return jsonify({'success': True, 'message': 'Emotion bias reset successfully'})
    except Exception as e:
        return jsonify({'error': f'Error resetting emotion bias: {str(e)}'})

@app.route('/api/adjust-emotion-weights', methods=['POST'])
def api_adjust_emotion_weights():
    """
    Adjust emotion detection weights
    """
    try:
        data = request.get_json()
        target_distribution = data.get('target_distribution') if data else None
        emotion_detector.detector.adjust_emotion_weights(target_distribution)
        return jsonify({'success': True, 'message': 'Emotion weights adjusted successfully'})
    except Exception as e:
        return jsonify({'error': f'Error adjusting emotion weights: {str(e)}'})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# SocketIO events
@socketio.on('connect')
def handle_connect():
    print(f'Client connected: {request.sid}')

@socketio.on('disconnect')
def handle_disconnect():
    print(f'Client disconnected: {request.sid}')

@socketio.on('process_frame')
def handle_frame_processing(data):
    """Handle real-time frame processing"""
    try:
        # Decode base64 image
        import base64
        image_data = data['image'].split(',')[1]  # Remove data:image/jpeg;base64, prefix
        image_bytes = base64.b64decode(image_data)
        
        # Process image
        results = emotion_detector.process_capture(image_bytes)
        
        # Send results back to client
        socketio.emit('frame_result', results)
        
    except Exception as e:
        print(f"Frame processing error: {e}")
        socketio.emit('frame_result', {'error': str(e)})

def draw_bounding_boxes_on_image(image, detections):
    """
    Draw bounding boxes dan labels pada image
    """
    image_copy = image.copy()
    
    for detection in detections:
        bbox = detection.get('bbox', [])
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            emotion = detection.get('emotion', 'Unknown')
            confidence = detection.get('emotion_confidence', 0)
            
            # Get emotion colors
            colors = get_emotion_colors(emotion)
            
            # Draw bounding box
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), colors['border'], 3)
            
            # Draw label background
            label_text = f"{emotion} ({confidence*100:.1f}%)"
            (label_width, label_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            cv2.rectangle(image_copy, (x1, y1 - label_height - 10), (x1 + label_width + 10, y1), colors['background'], -1)
            
            # Draw label text
            cv2.putText(image_copy, label_text, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['text'], 2)
            
            # Draw emotion emoji
            emoji = get_emotion_emoji(emotion)
            cv2.putText(image_copy, emoji, (x1 + label_width + 15, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors['border'], 2)
    
    return image_copy

def get_emotion_colors(emotion):
    """
    Get colors untuk emotion
    """
    color_map = {
        'Happy': {
            'border': (0, 255, 0),      # Green
            'background': (0, 255, 0, 0.9),
            'text': (0, 0, 0)           # Black
        },
        'Neutral': {
            'border': (255, 0, 0),      # Blue
            'background': (255, 0, 0, 0.9),
            'text': (255, 255, 255)     # White
        },
        'Sad': {
            'border': (128, 128, 128),  # Gray
            'background': (128, 128, 128, 0.9),
            'text': (255, 255, 255)     # White
        },
        'Angry': {
            'border': (0, 0, 255),      # Red
            'background': (0, 0, 255, 0.9),
            'text': (255, 255, 255)     # White
        },
        'Surprised': {
            'border': (255, 0, 255),    # Magenta
            'background': (255, 0, 255, 0.9),
            'text': (255, 255, 255)     # White
        }
    }
    
    return color_map.get(emotion, color_map['Neutral'])

def get_emotion_emoji(emotion):
    """
    Get emoji untuk emotion
    """
    emoji_map = {
        'Happy': '😊',
        'Neutral': '😐',
        'Sad': '😢',
        'Angry': '😠',
        'Surprised': '😲'
    }
    
    return emoji_map.get(emotion, '❓')

if __name__ == '__main__':
    print("🚀 Starting Simple Emotion Detection Flask App...")
    print("📱 Available routes:")
    print("   - / (Home)")
    print("   - /capture (Photo Capture)")
    print("   - /upload (Image Upload)")
    print("   - /record (Video Recording)")
    print("   - /realtime (Real-time Detection)")
    print("   - /emotion-summary (Temporal Analysis)")
    print("   - /reset-tracking (Reset Tracking)")
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000) 