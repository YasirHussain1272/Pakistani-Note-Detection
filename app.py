from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import pickle
import sys
import io

# Set encoding to UTF-8 for the environment
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['JSON_AS_ASCII'] = False  # Allow non-ASCII in JSON
IMG_SIZE = 128  # Image size for resizing

# Load the trained model and label binarizer
with open('currency_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

with open('label_binarizer.pkl', 'rb') as f:
    lb = pickle.load(f)

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Function to preprocess the image for prediction
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize to 128x128
    img = img / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension (1, 128, 128, 3)
    return img

# Predict the class of the currency note
def predict_image(image_path):
    img = preprocess_image(image_path)
    prediction = loaded_model.predict(img)  # Predict using the model
    predicted_class = lb.classes_[np.argmax(prediction)]  # Get class label
    return predicted_class

# Home route to display the upload form
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle the image upload and prediction
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Predict the uploaded image
        predicted_note = predict_image(filepath)
        predicted_note_safe = predicted_note.encode('utf-8').decode('utf-8')  # Ensure it's safe for rendering
        
        # Render the result and display the uploaded image
        return render_template('result.html', label=predicted_note_safe, image_path=filepath)

# Start the Flask app
if __name__ == "__main__":
    app.secret_key = "secret_key"
    app.run(debug=True)
