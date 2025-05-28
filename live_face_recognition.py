import os
import cv2
import numpy as np
import face_recognition

# Path to the known faces directory
KNOWN_FACES_DIR = os.path.join('face_site', 'known_faces')

# Load known face encodings
def load_known_faces():
    known_encodings = []
    known_names = []

    for name in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, name)
        if not os.path.isdir(person_dir):
            continue
        for filename in os.listdir(person_dir):
            img_path = os.path.join(person_dir, filename)
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(name)

    return known_encodings, known_names

def main():
    print("[INFO] Loading known faces...")
    known_encodings, known_names = load_known_faces()

    print("[INFO] Starting video stream...")
    video = cv2.VideoCapture(0)

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Resize for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_names[best_match_index]

            # Scale back up
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow("Live Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
