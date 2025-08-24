# File: app_unified_yolo.py
"""
Contoh implementasi Flask dengan Unified YOLO Detector
Menggunakan YOLO untuk Face Detection, Emotion Detection, dan Temporal Analysis
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_socketio import SocketIO
import cv2
import numpy as np
import os
import time
import base64
from src.yolo_unified_integration import FlaskYOLOIntegration

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
socketio = SocketIO(app, async_mode="threading")

# Membuat folder uploads jika belum ada
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize Unified YOLO Detector
print("🚀 Initializing Unified YOLO Detector...")
yolo_detector = FlaskYOLOIntegration('models/yolov8s.pt')  # Changed to yolov8s
print("✅ Unified YOLO Detector ready!")

# Emotion emojis dan colors
EMOTION_EMOJIS = {
    "Anger": "😡", "Contempt": "😏", "Disgust": "🤢", "Fear": "😱",
    "Happy": "😂", "Neutral": "😐", "Sad": "😢", "Surprised": "😲",
    "Unknown": "❓"
}

EMOTION_COLORS = {
    "Happy": "rgb(0, 255, 0)", "Anger": "rgb(255, 0, 0)", "Neutral": "rgb(0, 0, 255)",
    "Sad": "rgb(128, 128, 128)", "Contempt": "rgb(255, 255, 0)", "Disgust": "rgb(128, 0, 128)",
    "Fear": "rgb(255, 165, 0)", "Surprised": "rgb(255, 0, 255)", "Unknown": "rgb(255, 255, 255)"
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
    return render_template('realtime.html')

@app.route('/capture', methods=['POST'])
def process_capture():
    """
    Process captured photo menggunakan Unified YOLO Detector
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        confidence = float(request.form.get('confidence', 0.5))
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        # Process image dengan unified detector
        results = yolo_detector.process_capture(file.read())
        
        if 'error' in results:
            return jsonify(results)
        
        # Save processed image
        timestamp = int(time.time())
        filename = f"capture_{timestamp}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        # Convert results back to image untuk display
        # (Ini bisa dioptimasi lebih lanjut)
        cv2.imwrite(filepath, np.zeros((480, 640, 3), dtype=np.uint8))
        
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
    Process uploaded image menggunakan Unified YOLO Detector
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        confidence = float(request.form.get('confidence', 0.5))
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        # Process image dengan unified detector
        results = yolo_detector.process_upload(file)
        
        if 'error' in results:
            return jsonify(results)
        
        # Save original image
        timestamp = int(time.time())
        original_filename = f"upload_{timestamp}_{file.filename}"
        original_filepath = os.path.join(UPLOAD_FOLDER, original_filename)
        file.seek(0)  # Reset file pointer
        file.save(original_filepath)
        
        # Save processed image
        processed_filename = f"result_{timestamp}_{file.filename}"
        processed_filepath = os.path.join(UPLOAD_FOLDER, processed_filename)
        
        # Convert results back to image untuk display
        # (Ini bisa dioptimasi lebih lanjut)
        cv2.imwrite(processed_filepath, np.zeros((480, 640, 3), dtype=np.uint8))
        
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
    Process realtime frame menggunakan Unified YOLO Detector
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
        
        # Process frame dengan unified detector
        results = yolo_detector.detector.process_image(image)
        
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
def get_emotion_summary():
    """
    Get emotion summary dari temporal analysis
    """
    try:
        summary = yolo_detector.get_emotion_summary()
        return jsonify(summary)
    except Exception as e:
        return jsonify({'error': f'Error getting summary: {str(e)}'})

@app.route('/reset-tracking')
def reset_tracking():
    """
    Reset semua tracking data
    """
    try:
        yolo_detector.reset_tracking()
        return jsonify({'success': True, 'message': 'Tracking data reset'})
    except Exception as e:
        return jsonify({'error': f'Error resetting tracking: {str(e)}'})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# SocketIO events untuk realtime communication
@socketio.on('connect')
def handle_connect():
    print("Client connected:", request.sid)

@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected:", request.sid)

@socketio.on('process_frame')
def handle_process_frame(data):
    """
    Handle realtime frame processing via WebSocket
    """
    try:
        # Decode base64 image
        image_data = base64.b64decode(data['image'].split(',')[1])
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            socketio.emit('frame_result', {'error': 'Invalid image data'})
            return
        
        # Process frame dengan unified detector
        results = yolo_detector.detector.process_image(image)
        
        # Emit results back to client
        socketio.emit('frame_result', {
            'success': True,
            'results': results,
            'timestamp': time.time()
        })
        
    except Exception as e:
        socketio.emit('frame_result', {'error': f'Processing error: {str(e)}'})

if __name__ == '__main__':
    print("🚀 Starting Unified YOLO Flask App...")
    print("📱 Available routes:")
    print("   - / (Home)")
    print("   - /capture (Photo Capture)")
    print("   - /upload (Image Upload)")
    print("   - /record (Video Recording)")
    print("   - /realtime (Real-time Detection)")
    print("   - /emotion-summary (Temporal Analysis)")
    print("   - /reset-tracking (Reset Tracking)")
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000) 