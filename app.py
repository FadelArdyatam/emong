# File: app.py
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_socketio import SocketIO
import cv2
import numpy as np
import os
import time
import requests
import atexit
from src.model_loader import load_model
from src.emotion_detector import detect_emotion, draw_bounding_box
from config import MODEL_PATH, UPLOADS_DIR, GEMINI_API_KEY
from langdetect import detect, DetectorFactory

# Memastikan langdetect memberikan hasil yang konsisten
DetectorFactory.seed = 0

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
socketio = SocketIO(app, async_mode="threading")

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


# Load YOLO model
print("Loading YOLO model...")
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

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

EMOTION_COLORS_BGR = {
    "Happy": (0, 255, 0),
    "Anger": (0, 0, 255),
    "Neutral": (255, 0, 0),
    "Sad": (128, 128, 128),
    "Contempt": (0, 255, 255),
    "Disgust": (128, 0, 128),
    "Fear": (0, 165, 255),
    "Surprised": (255, 0, 255),
    "No face detected": (255, 255, 255),
    "Unknown": (255, 255, 255),
}

# Respons khusus berdasarkan emosi
EMOTION_PROMPTS = {
    "Happy": {
        "initial": "Haha, kamu kelihatan bahagia banget! 😄 Ada kabar seru apa nih? Ceritain ke aku dong! 🎉",
        "tone": "ceria, energik, suka bercanda, banyak emoji seperti 😄, 🎉, sapaan 'kamu-aku'",
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
            "message": "Tớ không kết nối được với API Gemini... 😓 Thử lại sau nha, giờ kể tớ nghe cậu đang nghĩ gì đi! 😊",
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
        prompt = f"You are a friendly chatbot acting like a close friend. Respond in Vietnamese with a {tone} tone. Use informal language, address the user as 'cậu' or 'bạn' as specified, and include emojis to make it lively. "
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

        print("Detecting emotions...")
        detections = detect_emotion(model, image, confidence_threshold)
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
                "emotion": e,
                "confidence": c,
                "emoji": EMOTION_EMOJIS.get(e, "❓"),
                "color": EMOTION_COLORS_CSS.get(e, "rgb(255, 255, 255)"),
            }
            for e, c, _ in detections
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

        print("Detecting emotions...")
        detections = detect_emotion(model, image, confidence_threshold)
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
                "emotion": e,
                "confidence": c,
                "emoji": EMOTION_EMOJIS.get(e, "❓"),
                "color": EMOTION_COLORS_CSS.get(e, "rgb(255, 255, 255)"),
            }
            for e, c, _ in detections
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

        # Tạo video đầu ra với codec webm
        result_filename = f"processed_{filename}"
        result_path = os.path.join(UPLOADS_DIR, result_filename)
        fourcc = cv2.VideoWriter_fourcc(*"VP80")
        out = cv2.VideoWriter(result_path, fourcc, fps, (width, height))
        if not out.isOpened():
            cap.release()
            print("Error: Failed to create output video")
            return jsonify({"error": "Failed to create output video"}), 500

        # Xử lý từng frame và vẽ bounding box
        emotions = {}
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % 30 == 0:
                detections = detect_emotion(model, frame, confidence_threshold)
                for e, c, _ in detections:
                    emotions[e] = max(emotions.get(e, 0), c)
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

        results = [
            {
                "emotion": e,
                "confidence": c,
                "emoji": EMOTION_EMOJIS.get(e, "❓"),
                "color": EMOTION_COLORS_CSS.get(e, "rgb(255, 255, 255)"),
            }
            for e, c in emotions.items()
        ]

        video_url = f"/uploads/{result_filename}"
        print(f"Sending response: video={video_url}, results={results}")
        return jsonify({"video": video_url, "results": results})
    else:
        return render_template("record.html")


@app.route("/realtime", methods=["GET"])
def realtime():
    return render_template("realtime.html")

#region: # Realtime emotion detection from webcam

@socketio.on("frame")
def handle_frame(data):
    print("Received frame from webcam...")
    confidence_threshold = data.get("confidence", 0.3)
    print(f"Confidence threshold: {confidence_threshold}")
    image_data = np.frombuffer(data["image"], np.uint8)
    frame = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    if frame is None:
        print("Error: Invalid frame received")
        socketio.emit(
            "result_frame",
            {
                "error": "Invalid frame",
                "results": [
                    {
                        "emotion": "Error",
                        "confidence": 0.0,
                        "emoji": "❓",
                        "color": "rgb(255, 0, 0)",
                    }
                ],
                "fps": 0,
            },
        )
        return

    print(f"Frame shape: {frame.shape}")
    detections = detect_emotion(model, frame, confidence_threshold)
    print(f"Detections: {detections}")

    results = [
        {
            "emotion": e,
            "confidence": c,
            "emoji": EMOTION_EMOJIS.get(e, "❓"),
            "color": EMOTION_COLORS_CSS.get(e, "rgb(255, 255, 255)"),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
        }
        for e, c, (x1, y1, x2, y2) in detections
    ]

    if not results:
        results = [
            {
                "emotion": "No face detected",
                "confidence": 0.0,
                "emoji": EMOTION_EMOJIS["No face detected"],
                "color": EMOTION_COLORS_CSS["No face detected"],
                "bbox": [0, 0, 0, 0],
            }
        ]

    primary_emotion = (
        max(results, key=lambda x: x["confidence"])["emotion"]
        if results
        else "No face detected"
    )

    print(f"Sending detection results: {results}")
    socketio.emit(
        "result_frame",
        {
            "results": results,
            "fps": data.get("fps", 0),
            "primary_emotion": primary_emotion,
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
