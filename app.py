# File: app.py
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_socketio import SocketIO
import cv2
import numpy as np
import os
import time
import requests
import atexit
import torch # Added for PyTorch model loading
from src.model_loader import load_model
# --- MODIFIED IMPORT ---
from src.emotion_detector import (
    load_known_faces,
    detect_emotions_and_recognize_faces,
    draw_bounding_box,
    get_emotion_colors,
)
from src.efficientnet_bilstm_model import EfficientNetBiLSTM # Added for new emotion model
from config import MODEL_PATH, UPLOADS_DIR, GEMINI_API_KEY, BASE_DIR

# Path to the trained emotion model (assuming it's in the 'models' directory)
EMOTION_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'emotion_efficientnet_bilstm_trained.pth')
from langdetect import detect, DetectorFactory

# Memastikan langdetect memberikan hasil yang konsisten
DetectorFactory.seed = 0

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
socketio = SocketIO(app, async_mode="eventlet", cors_allowed_origins="*")

# Thread-safe cache for real-time processing
import threading
recent_face_cache_realtime = []
face_sequence_buffers_realtime = {}
cache_lock = threading.Lock()

# Emotion tracking for analytics
emotion_history = []
emotion_history_lock = threading.Lock()

# Membuat folder uploads jika belum ada
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# Event SocketIO
@socketio.on("connect")
def handle_connect():
    print("Klien terhubung:", request.sid)


@socketio.on("disconnect")
def handle_disconnect():
    print("Klien terputus:", request.sid)


# --- LOAD MODELS AND FACES ON STARTUP ---
print("Loading YOLOv12s model for face detection...")
try:
    face_detector_model = load_model(MODEL_PATH) # MODEL_PATH is yolov12s.pt
except Exception as e:
    print(f"Error loading YOLOv12s model: {e}")
    exit(1)

print("Loading EfficientNet BiLSTM emotion model...")
try:
    # Define the number of emotion classes (must match your trained model)
    NUM_EMOTION_CLASSES = 7 # Adjust this if your model was trained with a different number of classes
    emotion_model = EfficientNetBiLSTM(num_emotion_classes=NUM_EMOTION_CLASSES)
    
    # Load model with better error handling
    if os.path.exists(EMOTION_MODEL_PATH):
        checkpoint = torch.load(EMOTION_MODEL_PATH, map_location=torch.device('cpu'))
        # Load only the parts that match the current model architecture
        model_dict = emotion_model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        emotion_model.load_state_dict(model_dict, strict=False)
        print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} layers from checkpoint")
    else:
        print(f"Warning: Model file not found at {EMOTION_MODEL_PATH}")
        print("Using randomly initialized model")
    
    emotion_model.eval() # Set to evaluation mode
    print("EfficientNet BiLSTM emotion model loaded successfully.")
except Exception as e:
    print(f"Error loading EfficientNet BiLSTM emotion model from {EMOTION_MODEL_PATH}: {e}")
    print("Continuing with basic functionality...")
    emotion_model = None

print("Loading known faces...")
known_face_encodings, known_face_names = load_known_faces('known_faces')
# --- END LOADING ---


# Emotion emojis and colors
EMOTION_EMOJIS = {
    "Anger": "😡",
    "Contempt": "😏",
    "Disgust": "🤢",
    "Fear": "😱",
    "Happy": "😂",
    "Neutral": "😐",
    "Sad": "😢",
    "Surprised": "😲",
    "No face detected": "❓",
    "Unknown": "❓",
}

EMOTION_COLORS_CSS = {
    "Happy": "rgb(0, 255, 0)",
    "Anger": "rgb(255, 0, 0)",
    "Neutral": "rgb(0, 0, 255)",
    "Sad": "rgb(128, 128, 128)",
    "Contempt": "rgb(255, 255, 0)",
    "Disgust": "rgb(128, 0, 128)",
    "Fear": "rgb(255, 165, 0)",
    "Surprised": "rgb(255, 0, 255)",
    "No face detected": "rgb(255, 255, 255)",
    "Unknown": "rgb(255, 255, 255)",
}

# Menggunakan fungsi dari emotion_detector untuk konsistensi
EMOTION_COLORS_BGR = get_emotion_colors()

# Respons khusus berdasarkan emosi
EMOTION_PROMPTS = {
    "Happy": {
        "initial": "Haha, kamu kelihatan bahagia banget! 😄 Ada kabar seru apa nih? Ceritain ke aku dong! 🎉",
        "tone": "ceria, energik, suka bercanda, banyak emoji seperti ��, 🎉, sapaan 'kamu-aku'",
    },
    "Anger": {
        "initial": "Waduh, kamu kenapa kelihatan marah begitu? 😣 Ada yang nggak beres ya? Cerita ke saya biar bisa bantu! 🤗",
        "tone": "penuh pengertian, lembut, perhatian, emoji seperti 🤗, 😣, sapaan 'kamu-saya'",
    },
    "Neutral": {
        "initial": "Hari ini kamu gimana? 😊 Kelihatannya santai banget ya? 🌟",
        "tone": "ramah, lembut, penasaran, emoji seperti 😊, 🌟, sapaan 'kamu-aku'",
    },
    "Sad": {
        "initial": "Kamu kok kelihatan sedih? 😢 Ada apa? Aku di sini kok, cerita aja ke aku! 💖",
        "tone": "menghibur, hangat, perhatian, emoji seperti 💖, 😢, sapaan 'kamu-aku'",
    },
    "Contempt": {
        "initial": "Hmm, kamu lagi nggak senang sama sesuatu ya? 😏 Ada yang mengganggu? Saya mau dengar ceritanya! 🤔",
        "tone": "penasaran, lembut, perhatian, emoji seperti 🤔, 😏, sapaan 'kamu-saya'",
    },
    "Disgust": {
        "initial": "Ih, kayaknya kamu lagi nggak suka sama sesuatu ya? 🤢 Ada hal yang bikin nggak nyaman? Ceritain ke aku! 😣",
        "tone": "penuh pengertian, penasaran, perhatian, emoji seperti 😣, 🤢, sapaan 'kamu-aku'",
    },
    "Fear": {
        "initial": "Waduh, kamu kok kelihatan takut? 😱 Ada yang bikin khawatir? Aku di sini temenin kamu! 🤗",
        "tone": "menghibur, melindungi, lembut, emoji seperti 🤗, 😱, sapaan 'kamu-aku'",
    },
    "Surprised": {
        "initial": "Wah, kamu kaget banget sampe matanya membulat gitu? 😲 Ada hal menarik apa nih? Ceritain ke aku! 🎉",
        "tone": "antusias, penasaran, ceria, emoji seperti 🎉, 😲, sapaan 'kamu-aku'",
    },
    "No face detected": {
        "initial": "Eh, aku nggak lihat kamu sama sekali! 😅 Kamu lagi sembunyi ya? Muncul dong, ngobrol sama aku! 😜",
        "tone": "ceria, penasaran, bercanda, emoji seperti 😅, 😜, sapaan 'kamu-aku'",
    },
    "Unknown": {
        "initial": "Hmm, aku belum bisa nebak perasaan kamu nih! 😕 Lagi mikirin apa? Ceritain ke aku! 😊",
        "tone": "ramah, penasaran, lembut, emoji seperti 😊, 😕, sapaan 'kamu-aku'",
    },
}


# English prompts for emotions (for responses in English)
EMOTION_PROMPTS_ENGLISH = {
    "Happy": {
        "initial": "OMG bro, u look so happy! 😍 What’s making u smile like that? Spill the tea! 🎉",
        "tone": "super chill, hype, playful, use lots of emojis like 😍, 🎉, teencode like 'u', 'bro', 'lol'",
    },
    "Anger": {
        "initial": "Whoa bro, u look kinda pissed! 😤 What’s got u so mad? Tell me, I gotchu! 🤗",
        "tone": "supportive, chill, caring, use emojis like 🤗, 😤, teencode like 'u', 'gotchu', 'bro'",
    },
    "Neutral": {
        "initial": "Hey bro, u seem chill today! 😎 How’s ur day going? Got any fun stuff to share? 🌟",
        "tone": "friendly, curious, relaxed, use emojis like 😎, 🌟, teencode like 'u', 'bro', 'ur'",
    },
    "Sad": {
        "initial": "Aww bro, u look so down... 🥺 What’s wrong? I’m here for u, let’s talk! 💖",
        "tone": "comforting, warm, caring, use emojis like 💖, 🥺, teencode like 'u', 'bro', 'let’s'",
    },
    "Contempt": {
        "initial": "Hmm, u look like u’re judging smth! 😏 What’s up? Spill it, I’m curious! 🤔",
        "tone": "curious, playful, chill, use emojis like 🤔, 😏, teencode like 'u', 'smth', 'bro'",
    },
    "Disgust": {
        "initial": "Eww bro, what’s making u look so grossed out? 🤢 Tell me, I wanna know! 😝",
        "tone": "curious, playful, chill, use emojis like 😝, 🤢, teencode like 'u', 'wanna', 'bro'",
    },
    "Fear": {
        "initial": "Oh no bro, u look kinda scared! 😱 What’s freaking u out? I’m here, talk to me! 🤗",
        "tone": "comforting, protective, chill, use emojis like 🤗, 😱, teencode like 'u', 'bro', 'freaking'",
    },
    "Surprised": {
        "initial": "Whoa bro, u look so shocked! 😲 What’s got u like that? Tell me quick! 🎉",
        "tone": "excited, curious, playful, use emojis like 🎉, 😲, teencode like 'u', 'bro', 'quick'",
    },
    "No face detected": {
        "initial": "Yo bro, I can’t see u! 😅 U hiding or what? Show ur face and chat with me! 😜",
        "tone": "playful, curious, chill, use emojis like 😅, 😜, teencode like 'u', 'ur', 'bro'",
    },
    "Unknown": {
        "initial": "Hmm, I can’t tell how u’re feeling, bro! 🤔 What’s on ur mind? Tell me! 😊",
        "tone": "friendly, curious, chill, use emojis like 😊, 🤔, teencode like 'u', 'ur', 'bro'",
    },
}


def get_gemini_response(emotion, user_message=None):
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_ACTUAL_GEMINI_API_KEY":
        print("GEMINI_API_KEY is not set or invalid.")
        return {
            "message": "Aku tidak bisa terhubung ke API Gemini... 😓 Coba lagi nanti ya, sekarang cerita ke aku, kamu lagi mikirin apa! 😊",
            "status": "error",
        }

    # Mendeteksi bahasa dari pesan pengguna
    language = "id"  # Default adalah bahasa Indonesia
    if user_message:
        try:
            language = detect(user_message)
        except Exception as e:
            print(f"Error detecting language: {e}")
            language = "id"  # Fallback to Indonesia if detection fails

    # Memilih prompt berdasarkan bahasa
    if language == "id":
        prompt_data = EMOTION_PROMPTS.get(emotion, EMOTION_PROMPTS["Neutral"])
        tone = prompt_data["tone"]
        initial_message = prompt_data["initial"]
        prompt = f"Lo itu chatbot yang friendly banget, berperan kayak temen deket. Jawab pake bahasa Indonesia dengan tone {tone}. Pake bahasa santai ala anak Jaksel, manggil usernya 'lo' atau 'lu', dan selipin emoji biar vibes-nya hidup "
    else:
        prompt_data = EMOTION_PROMPTS_ENGLISH.get(
            emotion, EMOTION_PROMPTS_ENGLISH["Neutral"]
        )
        tone = prompt_data["tone"]
        initial_message = prompt_data["initial"]
        prompt = f"You are a friendly chatbot acting like a close friend. Respond in English with a {tone} tone. Use informal language, address the user as 'bro' or 'u', and include emojis to make it lively. Use teencode like 'u', 'ur', 'lol', 'smth', etc. to sound casual and friendly. "

    if user_message:
        prompt += f"The user said: '{user_message}'. Respond to their message naturally, keeping the {tone} tone and addressing them appropriately."
    else:
        prompt += f"Start the conversation with: '{initial_message}'."

    try:
        print(f"Calling Gemini API with prompt: {prompt}")
        response = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent",
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.9,
                    "topK": 40,
                    "topP": 0.95,
                    "maxOutputTokens": 1024,
                },
            },
            params={"key": GEMINI_API_KEY},
        )
        response_data = response.json()
        print(f"Gemini API response: {response_data}")

        if "error" in response_data:
            print(f"Gemini API error: {response_data['error']}")
            return {
                "message": (
                    "Aku mengalami masalah saat terhubung ke API Gemini... 😓 Ceritain sesuatu yang seru dulu deh sambil aku coba lagi! 😊"
                    if language == "id"
                    else "Oops, I can’t connect to the API right now... 😓 Tell me smth fun while I try again, bro! 😊"
                ),
                "status": "error",
            }

        return {
            "message": response_data["candidates"][0]["content"]["parts"][0]["text"],
            "status": "success",
        }

    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return {
            "message": (
                "Aku kena sedikit masalah nih... 😅 Gimana kalau kamu ceritain harimu? Aku penasaran banget! 😜"
                if language == "id"
                else "Oops, I hit a lil snag... 😅 Tell me how ur day’s going, I’m super curious! 😜"
            ),
            "status": "error",
        }


# region: # Routes
@app.route("/")
def index():
    print("Rendering index.html...")
    return render_template("index.html")


@app.route("/uploads/<filename>")
def serve_uploaded_file(filename):
    print(f"Serving file: {filename} from {UPLOADS_DIR}")
    return send_from_directory(UPLOADS_DIR, filename)

#region: # Upload image and detect emotion

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        print("Received upload request...")
        if "file" not in request.files:
            print("Error: No file uploaded")
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        confidence_threshold = float(request.form.get("confidence", 0.05))
        print(f"Confidence threshold: {confidence_threshold}")
        if not file.mimetype.startswith("image"):
            print("Error: Not an image file")
            return jsonify({"error": "Please upload an image file"}), 400

        filename = f"upload_{int(time.time())}_{file.filename}"
        file_path = os.path.join(UPLOADS_DIR, filename)
        print(f"Saving uploaded file to: {file_path}")
        file.save(file_path)

        print("Reading image...")
        image = cv2.imread(file_path)
        if image is None:
            print("Error: Failed to read image")
            return jsonify({"error": "Failed to read image"}), 500

        print("Detecting emotions and recognizing faces...")
        recent_face_cache = [] # Initialize cache for this single image request
        face_sequence_buffers = {} # Initialize sequence buffers for this single image request
        detections = detect_emotions_and_recognize_faces(face_detector_model, emotion_model, image, known_face_encodings, known_face_names, recent_face_cache, face_sequence_buffers, confidence_threshold)
        print(f"Detections: {detections}")

        print("Drawing bounding boxes...")
        result_image = draw_bounding_box(image.copy(), detections, EMOTION_COLORS_BGR)
        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        result_filename = f"result_{filename}"
        result_path = os.path.join(UPLOADS_DIR, result_filename)
        print(f"Saving result image to: {result_path}")
        success = cv2.imwrite(
            result_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
        )
        if not success:
            print("Error: Failed to save result image")
            return jsonify({"error": "Failed to save result image"}), 500

        if not os.path.exists(result_path):
            print("Error: Result image not found after saving")
            return jsonify({"error": "Result image not found"}), 500

        results = [
            {
                "name": name,
                "emotion": e,
                "confidence": c,
                "emoji": EMOTION_EMOJIS.get(e, "❓"),
                "color": EMOTION_COLORS_CSS.get(e, "rgb(255, 255, 255)"),
            }
            for name, e, c, _ in detections
        ]
        image_url = f"/uploads/{result_filename}"
        print(f"Sending response: image={image_url}, results={results}")
        return jsonify({"image": image_url, "results": results})
    else:
        return render_template("upload.html")


@app.route("/capture", methods=["GET", "POST"])
def capture():
    if request.method == "POST":
        print("Received capture request...")
        if "file" not in request.files:
            print("Error: No file uploaded")
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        confidence_threshold = float(request.form.get("confidence", 0.05))
        print(f"Confidence threshold: {confidence_threshold}")
        if not file.mimetype.startswith("image"):
            print("Error: Not an image file")
            return jsonify({"error": "Please upload an image file"}), 400

        filename = f"capture_{int(time.time())}.jpg"
        file_path = os.path.join(UPLOADS_DIR, filename)
        print(f"Saving captured file to: {file_path}")
        file.save(file_path)

        print("Reading image...")
        image = cv2.imread(file_path)
        if image is None:
            print("Error: Failed to read image")
            return jsonify({"error": "Failed to read image"}), 500

        print("Detecting emotions and recognizing faces...")
        recent_face_cache = [] # Initialize cache for this single image request
        face_sequence_buffers = {} # Initialize sequence buffers for this single image request
        detections = detect_emotions_and_recognize_faces(face_detector_model, emotion_model, image, known_face_encodings, known_face_names, recent_face_cache, face_sequence_buffers, confidence_threshold)
        print(f"Detections: {detections}")

        print("Drawing bounding boxes...")
        result_image = draw_bounding_box(image.copy(), detections, EMOTION_COLORS_BGR)
        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        result_filename = f"result_{filename}"
        result_path = os.path.join(UPLOADS_DIR, result_filename)
        print(f"Saving result image to: {result_path}")
        success = cv2.imwrite(
            result_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
        )
        if not success:
            print("Error: Failed to save result image")
            return jsonify({"error": "Failed to save result image"}), 500

        if not os.path.exists(result_path):
            print("Error: Result image not found after saving")
            return jsonify({"error": "Result image not found"}), 500

        results = [
            {
                "name": name,
                "emotion": e,
                "confidence": c,
                "emoji": EMOTION_EMOJIS.get(e, "❓"),
                "color": EMOTION_COLORS_CSS.get(e, "rgb(255, 255, 255)"),
            }
            for name, e, c, _ in detections
        ]
        image_url = f"/uploads/{result_filename}"
        print(f"Sending response: image={image_url}, results={results}")
        return jsonify({"image": image_url, "results": results})
    else:
        return render_template("capture.html")

#region: # Record video and process it

@app.route("/record", methods=["GET", "POST"])
def record():
    if request.method == "POST":
        print("Received record request...")
        if "file" not in request.files: 
            print("Error: No file uploaded")
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        confidence_threshold = float(request.form.get("confidence", 0.05))
        print(f"Confidence threshold: {confidence_threshold}")
        if not file.mimetype.startswith("video"):
            print("Error: Not a video file")
            return jsonify({"error": "Please upload a video file"}), 400

        filename = f"record_{int(time.time())}.webm"
        file_path = os.path.join(UPLOADS_DIR, filename)
        print(f"Saving recorded file to: {file_path}")
        file.save(file_path)

        print("Processing video...")
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            print("Error: Failed to open video")
            return jsonify({"error": "Failed to open video"}), 500

        # Lấy thông tin video
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Tạo video đầu ra dengan codec webm
        result_filename = f"processed_{filename}"
        result_path = os.path.join(UPLOADS_DIR, result_filename)
        fourcc = cv2.VideoWriter_fourcc(*"VP80")
        out = cv2.VideoWriter(result_path, fourcc, fps, (width, height))
        if not out.isOpened():
            cap.release()
            print("Error: Failed to create output video")
            return jsonify({"error": "Failed to create output video"}), 500

        detected_people = {}
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Proses setiap beberapa frame untuk efisiensi
            if frame_count % 15 == 0: # Proses lebih sering untuk video
                # Initialize cache for video processing session
                if 'recent_face_cache_record' not in locals():
                    recent_face_cache_record = []
                if 'face_sequence_buffers_record' not in locals(): # New: Initialize sequence buffers for video
                    face_sequence_buffers_record = {}
                detections = detect_emotions_and_recognize_faces(face_detector_model, emotion_model, frame, known_face_encodings, known_face_names, recent_face_cache_record, face_sequence_buffers_record, confidence_threshold)
                for name, emotion, conf, _ in detections:
                    if name not in detected_people:
                        detected_people[name] = {}
                    detected_people[name][emotion] = max(detected_people[name].get(emotion, 0), conf)
                
                result_frame = draw_bounding_box(
                    frame.copy(), detections, EMOTION_COLORS_BGR
                )
                out.write(result_frame)
            else:
                out.write(frame)
            frame_count += 1

        cap.release()
        out.release()

        if not os.path.exists(result_path):
            print("Error: Processed video not found")
            return jsonify({"error": "Processed video not found"}), 500

        # Siapkan hasil akhir berdasarkan orang yang terdeteksi
        final_results = []
        for name, emotions in detected_people.items():
            # Cari emosi dominan untuk setiap orang
            if emotions:
                dominant_emotion = max(emotions, key=emotions.get)
                confidence = emotions[dominant_emotion]
                final_results.append({
                    "name": name,
                    "emotion": dominant_emotion,
                    "confidence": confidence,
                    "emoji": EMOTION_EMOJIS.get(dominant_emotion, "❓"),
                    "color": EMOTION_COLORS_CSS.get(dominant_emotion, "rgb(255, 255, 255)"),
                })

        video_url = f"/uploads/{result_filename}"
        print(f"Sending response: video={video_url}, results={final_results}")
        return jsonify({"video": video_url, "results": final_results})
    else:
        return render_template("record.html")


@app.route("/realtime", methods=["GET"])
def realtime():
    return render_template("realtime.html")

@app.route("/api/emotion-data", methods=["GET"])
def get_emotion_data():
    """Get emotion data for analytics"""
    with emotion_history_lock:
        # Return last 100 emotion records
        recent_emotions = emotion_history[-100:] if len(emotion_history) > 100 else emotion_history
        
        # Calculate emotion distribution
        emotion_counts = {}
        for record in recent_emotions:
            emotion = record.get('emotion', 'Unknown')
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        return jsonify({
            'recent_emotions': recent_emotions,
            'emotion_distribution': emotion_counts,
            'total_records': len(emotion_history)
        })

#region: # Realtime emotion detection from webcam

@socketio.on("frame")
def handle_frame(data):
    confidence_threshold = data.get("confidence", 0.3)
    
    # Fix: Handle both list and bytes data
    if isinstance(data["image"], list):
        image_data = np.array(data["image"], dtype=np.uint8)
    else:
        image_data = np.frombuffer(data["image"], np.uint8)
    
    frame = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    if frame is None:
        return

    # Use thread-safe cache for real-time stream
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

    # Track emotion history for analytics
    with emotion_history_lock:
        for result in results:
            if result['emotion'] not in ['No face detected', 'Unknown']:
                emotion_record = {
                    'timestamp': time.time(),
                    'name': result['name'],
                    'emotion': result['emotion'],
                    'confidence': result['confidence'],
                    'emoji': result['emoji']
                }
                emotion_history.append(emotion_record)
                
                # Keep only last 1000 records to prevent memory issues
                if len(emotion_history) > 1000:
                    emotion_history.pop(0)

    # Jika tidak ada deteksi, kirim pesan khusus
    if not results:
        results.append({
            "name": "Unknown",
            "emotion": "No face detected",
            "confidence": 0.0,
            "emoji": EMOTION_EMOJIS["No face detected"],
            "color": EMOTION_COLORS_CSS["No face detected"],
            "bbox": [0, 0, 0, 0],
        })

    socketio.emit(
        "result_frame",
        {
            "results": results,
            "fps": data.get("fps", 0),
        },
    )

#region # Chat functionality

@socketio.on("chat_message")
def handle_chat_message(data):
    print("Received chat message:", data)
    user_message = data.get("message")
    emotion = data.get("emotion", "Neutral")

    socketio.emit("chat_responding", {"status": "responding"})
    bot_response = get_gemini_response(emotion, user_message)

    socketio.emit("chat_response", bot_response)


@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


if __name__ == "__main__":
    print("Starting Flask server...")
    socketio.run(app, debug=True)
#endregion