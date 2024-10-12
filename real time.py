import cv2
import numpy as np
import pickle
import sys
import io

# Change the standard output to UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load the model and label encoder
with open('currency_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

with open('label_binarizer.pkl', 'rb') as f:
    lb = pickle.load(f)

# Constants
IMG_SIZE = 128  # The size to which the images will be resized

# Function to predict the currency note from an image
def predict_image(image):
    img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))  # Resize to 128x128
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Predict using the loaded model
    prediction = loaded_model.predict(img)
    predicted_class = lb.classes_[np.argmax(prediction)]
    return predicted_class

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Main loop for real-time detection
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Predict the currency note in the current frame
    detected_class = predict_image(frame)

    # Display the result on the frame, ensuring it is encoded correctly
    try:
        cv2.putText(frame, f'Detected: {detected_class}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    except Exception as e:
        print(f"Error displaying text: {e}")

    # Show the frame
    cv2.imshow('Currency Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
