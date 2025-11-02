import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from datetime import datetime
import os

# --- 1. Initialization ---

# Load your trained age/gender model
print("Loading age/gender model...")
age_gender_model = tf.keras.models.load_model('age_gender_model.h5')
IMAGE_SIZE = 224 # Must match the training size

# Load the OpenCV face detector
print("Loading face detector model...")
face_net = cv2.dnn.readNetFromCaffe(
    'deploy.prototxt',
    'res10_300x300_ssd_iter_140000.caffemodel'
)

# List to store log data
log_data = []
LOG_FILENAME = "senior_citizen_log.csv"

# Start video capture
# Use 0 for webcam. If it doesn't work, try 1.
# Or, use a video file path: 'my_video.mp4'
cap = cv2.VideoCapture(0) 

print("Starting webcam feed...")

# --- 2. Main Loop (Frame-by-Frame Processing) ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    (h, w) = frame.shape[:2]
    
    # Preprocess the frame for face detection
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    # Get face detections
    face_net.setInput(blob)
    detections = face_net.forward()

    # --- 3. Loop Over Detections ---
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.6: # Filter weak detections
            # Get bounding box coordinates
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Ensure box is within frame boundaries
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w - 1, endX)
            endY = min(h - 1, endY)
            
            if startX >= endX or startY >= endY:
                continue

            # --- 4. Classify Age & Gender ---
            try:
                # Extract the face Region of Interest (ROI)
                face = frame[startY:endY, startX:endX]
                
                # Preprocess the face for your model
                # 1. Convert BGR (OpenCV) to RGB (Keras)
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                # 2. Resize to 224x224
                processed_face = cv2.resize(face_rgb, (IMAGE_SIZE, IMAGE_SIZE))
                # 3. Apply MobileNetV2 preprocessing
                processed_face = preprocess_input(processed_face)
                # 4. Add batch dimension
                processed_face = np.expand_dims(processed_face, axis=0)

                # Predict age and gender
                # The model returns a list: [age_output, gender_output]
                predictions = age_gender_model.predict(processed_face)
                
                age = int(predictions[0][0][0])
                gender_prob = predictions[1][0][0]
                
                gender = "Female" if gender_prob > 0.5 else "Male"

                # --- 5. Apply Logic & Log Data ---
                label = ""
                color = (0, 255, 0) # Green for non-senior

                if age > 60:
                    label = f"Senior Citizen: {gender} ({age})"
                    color = (0, 0, 255) # Red for senior
                    
                    # Log the data
                    log_data.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "age": age,
                        "gender": gender,
                        "confidence": f"{gender_prob*100:.2f}%"
                    })
                else:
                    label = f"{gender} ({age})"

                # Draw the label and bounding box
                cv2.putText(frame, label, (startX, startY - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            except Exception as e:
                print(f"Error processing face: {e}")
                pass # Skip this face if preprocessing fails

    # --- 6. Display the Output ---
    cv2.imshow("Senior Citizen Detection System", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 7. Cleanup & Save Log ---
print("Shutting down...")
cap.release()
cv2.destroyAllWindows()

# Save the log file to CSV
if len(log_data) > 0:
    df = pd.DataFrame(log_data)
    df.to_csv(LOG_FILENAME, index=False)
    print(f"Log file saved to {LOG_FILENAME}")
else:
    print("No senior citizens were detected.")