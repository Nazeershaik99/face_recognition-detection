import os
from flask import Flask, request, render_template, Response, redirect, url_for
import cv2
import face_recognition

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join('static', 'images')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load known faces from known_faces folder
def load_known_faces():
    known_faces_dir = 'face_site/known_faces'
    known_encodings = []
    known_names = []

    for name in os.listdir(known_faces_dir):
        person_dir = os.path.join(known_faces_dir, name)
        if not os.path.isdir(person_dir):
            continue

        for filename in os.listdir(person_dir):
            filepath = os.path.join(person_dir, filename)
            image = face_recognition.load_image_file(filepath)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(name)
    return known_encodings, known_names

known_encodings, known_names = load_known_faces()

def predict_faces(image_path):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]

        cv2.rectangle(image_bgr, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image_bgr, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_' + os.path.basename(image_path))
    cv2.imwrite(output_path, image_bgr)
    return output_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    filename = file.filename
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    processed_image_path = predict_faces(file_path)
    return render_template('result.html', image_path=processed_image_path)

# Live camera feed with face recognition
def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_names[first_match_index]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/live_camera')
def live_camera():
    return render_template('live_camera.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
