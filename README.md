# 👁️‍🗨️ Face Recognition System using Flask and OpenCV

This project is a web-based Face Recognition System built with **Flask**, **OpenCV**, and **face_recognition** library. It allows users to either upload a photo for recognition or use the **live camera** to detect and recognize known faces in real-time.

---

## 🚀 Features

- ✅ Upload an image and recognize known faces.
- ✅ Live face recognition using a webcam.
- ✅ Easy UI with Tailwind CSS.
- ✅ Real-time bounding boxes and names overlayed on faces.
- ✅ Expandable known faces dataset.

---

## 🗂️ Project Structure

face_recognition/
├── app.py # Main Flask application
├── live_recognizer.py # Script for live camera face recognition
├── face_site/
│ └── known_faces/ # Subfolders named after people containing their face images
├── templates/
│ ├── index.html # Homepage with upload interface
│ └── live_recognition.html # Live recognition launch page
└── static/ # (Optional) Static files like CSS/images

yaml
Copy
Edit

---

## 📦 Requirements

Install dependencies using pip:

```bash
pip install flask opencv-python face_recognition numpy
Note: On Windows, you might need to install CMake and dlib first for face_recognition to work properly.

🧠 How It Works
known_faces/ contains subfolders (names of people), each with one or more images.

These images are encoded using face_recognition and stored in memory.

When an image is uploaded or live camera is used, it compares the faces with the known ones.

💻 Usage
1. Prepare the Dataset
Place images inside:

bash
Copy
Edit
face_site/known_faces/PersonName/image1.jpg
face_site/known_faces/PersonName/image2.jpg
2. Run the Flask App
bash
Copy
Edit
python app.py
Visit: http://127.0.0.1:5000

3. Upload Mode
Choose an image from your device.

Click "Start the Process" to perform face recognition.

4. Live Camera Mode
Go to http://127.0.0.1:5000/live

Click "Start Live Recognition"

A separate OpenCV window opens.

Press q to close the camera.

📸 Example
Upload:

Live Recognition:

🛠️ Future Enhancements
Store face recognition results in a database.

Add login system for secured access.

Add confidence score display.

🧑‍💻 Author
Made with ❤️ by [Your Name]

📄 License
This project is open-source and free to use under the MIT License.

yaml
Copy
Edit

---

Let me know if you want to include screenshots or GitHub Actions, or convert this into a GitHub Wiki page.
