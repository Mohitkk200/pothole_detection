from flask import Flask, render_template, request, Response
import os
import cv2
import time
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
MODEL_PATH = 'models/yolo11n.pt'

ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

model = None

def get_model():
    global model
    if model is None:
        model = YOLO(MODEL_PATH)
    return model

def allowed_file(filename, allowed_set):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_set

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'media' not in request.files:
        return "No file uploaded", 400

    file = request.files['media']
    if file.filename == '':
        return "No selected file", 400

    filename = secure_filename(file.filename)
    ext = filename.rsplit('.', 1)[1].lower()

    if ext not in ALLOWED_IMAGE_EXTENSIONS and ext not in ALLOWED_VIDEO_EXTENSIONS:
        return "Unsupported file type", 400

    upload_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(upload_path)

    if ext in ALLOWED_IMAGE_EXTENSIONS:
        return handle_image(upload_path, filename)
    else:
        return handle_video(upload_path, filename)

def handle_image(path, filename):
    model = get_model()
    results = model(path, device='cpu')
    annotated_img = results[0].plot()
    pothole_count = len(results[0].boxes) if results[0].boxes else 0

    result_img_path = os.path.join(RESULT_FOLDER, filename)
    cv2.imwrite(result_img_path, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
    display_path = result_img_path.replace('static/', '')

    return render_template('result.html', media_type='image', media_path=display_path, pothole_count=pothole_count)

def handle_video(path, filename):
    return render_template('result.html', media_type='video_stream', video_name=filename)

def get_frame(video_path):
    model = get_model()
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.resize(frame, (640, 360))  # Resize to reduce memory
        results = model(frame[..., ::-1], device='cpu')
        annotated = results[0].plot()

        ret, buffer = cv2.imencode('.jpg', annotated)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n\r\n')
        time.sleep(0.03)

    cap.release()
    cv2.destroyAllWindows()

@app.route('/video_feed/<filename>')
def video_feed(filename):
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    return Response(get_frame(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

# ✅ DO NOT include app.run() — Render uses gunicorn to run the app!
