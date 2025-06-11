from flask import Flask, render_template, request, Response, send_from_directory
import os
from werkzeug.utils import secure_filename
import cv2
import time

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
RESULT_FOLDER = 'static/results/'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov'}

MODEL_PATH = 'models/yolo11n.pt'

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

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(upload_path)

    if ext in ALLOWED_IMAGE_EXTENSIONS:
        return handle_image(upload_path, filename)
    elif ext in ALLOWED_VIDEO_EXTENSIONS:
        return handle_video(upload_path, filename)
    else:
        return "Unsupported file type", 400

def handle_image(path, filename):
    from ultralytics import YOLO
    model = YOLO(MODEL_PATH)
    results = model(path, device='cpu')

    annotated_img = results[0].plot()
    pothole_count = len(results[0].boxes) if results[0].boxes is not None else 0

    os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
    result_img_path = os.path.join(app.config['RESULT_FOLDER'], filename)
    cv2.imwrite(result_img_path, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
    display_path = result_img_path.replace('static/', '')

    return render_template('result.html', media_type='image', media_path=display_path, pothole_count=pothole_count)

def handle_video(path, filename):
    return render_template('result.html', media_type='video_stream', video_name=filename)

def get_frame(video_path):
    from ultralytics import YOLO
    model = YOLO(MODEL_PATH)
    video = cv2.VideoCapture(video_path)
    while True:
        success, frame = video.read()
        if not success:
            break
        results = model(frame[..., ::-1], device='cpu')
        annotated = results[0].plot()
        ret, jpeg = cv2.imencode('.jpg', annotated)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        time.sleep(0.03)

@app.route('/video_feed/<filename>')
def video_feed(filename):
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return Response(get_frame(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
