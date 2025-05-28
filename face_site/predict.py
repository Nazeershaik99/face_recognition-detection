import os
import cv2
import face_recognition
import numpy as np

# Define directories for known faces and processed output
KNOWN_FACES_DIR = "known_faces"
OUTPUT_DIR = "static/processed"

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Utility function to load an image and convert it to the required format
def load_and_convert_image(image_path):
    """
    Loads an image from the given path and converts it to an 8-bit, 3-channel BGR
    format, then also returns an RGB version suitable for face_recognition.

    Args:
        image_path (str): The path to the image file.

    Returns:
        tuple: A tuple containing:
            - bgr_img (numpy.ndarray): The image in BGR format (for OpenCV drawing).
            - rgb_img (numpy.ndarray): The image in RGB format (for face_recognition).

    Raises:
        ValueError: If the image cannot be read or has an unsupported number of channels.
    """
    # Read the image as is, preserving its original number of channels and depth
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"[ERROR] Could not read image: {image_path}")

    # Ensure the image data type is 8-bit unsigned integer (np.uint8)
    # If the image is float (e.g., 0.0-1.0 range), scale it to 0-255
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = (img * 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        # For other unusual dtypes, attempt to convert directly to uint8
        img = img.astype(np.uint8)

    # Handle different channel configurations to ensure a 3-channel BGR image
    if len(img.shape) == 2:
        # Grayscale image: convert to BGR (3 channels)
        bgr_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        # Image with alpha channel (e.g., RGBA or BGRA): remove alpha and convert to BGR
        bgr_img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.shape[2] == 3:
        # Already a 3-channel BGR image, no conversion needed for BGR output
        bgr_img = img
    else:
        # Raise an error for unsupported channel configurations
        raise ValueError(f"[ERROR] Unsupported number of channels ({img.shape[2]}) for image: {image_path}")

    # Convert the BGR image to RGB format, which is required by face_recognition
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    return bgr_img, rgb_img

# Function to load known faces from the 'known_faces' directory
def load_known_faces():
    """
    Loads face encodings and names from images in the KNOWN_FACES_DIR.
    Each subfolder in KNOWN_FACES_DIR is treated as a person's name.

    Returns:
        tuple: A tuple containing:
            - known_encodings (list): A list of face encodings for known individuals.
            - known_names (list): A list of names corresponding to the known encodings.
    """
    known_encodings = []
    known_names = []

    # Iterate through each subfolder (person's name) in the known faces directory
    for name in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, name)
        # Skip if it's not a directory
        if not os.path.isdir(person_dir):
            continue

        # Iterate through each image file within the person's directory
        for filename in os.listdir(person_dir):
            path = os.path.join(person_dir, filename)
            try:
                # Load and convert the image, getting the RGB version for face_recognition
                _, rgb_image = load_and_convert_image(path)
                # Get face encodings from the image
                encodings = face_recognition.face_encodings(rgb_image)
                if encodings:
                    # If a face is found, add its encoding and the person's name
                    known_encodings.append(encodings[0])
                    known_names.append(name)
                else:
                    print(f"[WARNING] No face found in image: {path}")
            except ValueError as e:
                # Catch and report errors during image loading/conversion
                print(f"[ERROR] Skipping image {path} due to: {e}")
            except Exception as e:
                # Catch any other unexpected errors
                print(f"[ERROR] An unexpected error occurred while processing {path}: {e}")
    return known_encodings, known_names

# Function to predict faces in a given image and draw bounding boxes
def predict_faces(image_path):
    """
    Detects faces in the given image, compares them against known faces,
    draws bounding boxes and names, and saves the processed image.

    Args:
        image_path (str): The path to the image file to process.

    Returns:
        str: The path to the processed output image, or None if the input
             image could not be processed.
    """
    # Load known faces and their encodings
    known_encodings, known_names = load_known_faces()

    # Warn if no known faces were loaded, as all detected faces will be 'Unknown'
    if not known_encodings:
        print("[WARNING] No known faces loaded. All detected faces will be labeled 'Unknown'.")

    try:
        # Load and convert the input image for prediction
        bgr_img, rgb_img = load_and_convert_image(image_path)
    except ValueError as e:
        print(f"[ERROR] Could not process input image {image_path}: {e}")
        return None  # Return None if the input image cannot be processed

    # Find all face locations and encodings in the input image
    face_locations = face_recognition.face_locations(rgb_img)
    encodings = face_recognition.face_encodings(rgb_img, face_locations)

    # Iterate through each detected face
    for (top, right, bottom, left), face_encoding in zip(face_locations, encodings):
        name = "Unknown" # Default name for unknown faces

        # Only attempt to compare if there are known faces loaded
        if known_encodings:
            # Compare the detected face with all known faces
            matches = face_recognition.compare_faces(known_encodings, face_encoding)

            # Find the best match if there are any matches
            if True in matches:
                # Calculate face distances to find the closest known face
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                best_match_index = np.argmin(face_distances) # Index of the closest known face

                # If the closest face is indeed a match, assign its name
                if matches[best_match_index]:
                    name = known_names[best_match_index]

        # Draw a rectangle around the detected face
        cv2.rectangle(bgr_img, (left, top), (right, bottom), (0, 255, 0), 2)

        # Get text size to create a background rectangle for better readability
        text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        # Draw a filled black rectangle behind the text
        cv2.rectangle(bgr_img, (left, top - text_size[1] - 10), (left + text_size[0], top), (0, 0, 0), cv2.FILLED)
        # Put the name text on the image
        cv2.putText(bgr_img, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Construct the output path for the processed image
    result_path = os.path.join(OUTPUT_DIR, os.path.basename(image_path))
    # Save the processed image
    cv2.imwrite(result_path, bgr_img)
    return result_path