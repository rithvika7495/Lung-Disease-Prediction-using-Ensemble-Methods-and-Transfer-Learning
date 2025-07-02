from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import os
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
from ensemble_model import load
from prediction import preprocess_image, get_gradcam_heatmap, display_gradcam, predict_disease, disease_info
from functools import lru_cache

app = Flask(__name__)

MEGABYTE = (2 ** 10) ** 2
app.config['MAX_CONTENT_LENGTH'] = None
app.config['MAX_FORM_MEMORY_SIZE'] = 50 * MEGABYTE

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'dcm'}

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

GRADCAM_FOLDER = 'gradcam_images'
app.config['GRADCAM_FOLDER'] = GRADCAM_FOLDER

if not os.path.exists(GRADCAM_FOLDER):
    os.makedirs(GRADCAM_FOLDER, exist_ok=True)

# Lazy load ensemble model
weights_path = r"models\ensemble_model_weights.keras"
w1 = r"models\inceptionresnetv2_model_weights.keras"
w2 = r"models\mobilenet_model_weights.keras"
w3 = r"models\densenet_model_weights.keras"
w4 = r"models\efficientnetb2_model_weights.keras"
w5 = r"models\googlenet_model_weights.keras"

@lru_cache(maxsize=1)
def get_ensemble_model():
    return load(weights_path,w1,w2,w3,w4,w5)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/gradcam_images/<filename>')
def gradcam_images(filename):
    return send_from_directory(app.config['GRADCAM_FOLDER'], filename)

def process_image(filepath):
    ensemble_model = get_ensemble_model()
    img_array = preprocess_image(filepath)
    heatmap = get_gradcam_heatmap(ensemble_model, img_array, 'mixed10')
    
    gradcam_image_path = os.path.join(app.config['GRADCAM_FOLDER'], os.path.splitext(os.path.basename(filepath))[0] + '_gradcam.png')
    
    img = tf.keras.preprocessing.image.load_img(filepath)
    img = tf.keras.preprocessing.image.img_to_array(img)
    display_gradcam(img, heatmap, save_path=gradcam_image_path)

    if os.path.exists(gradcam_image_path):
        print(f"Grad-CAM image saved successfully: {gradcam_image_path}")
    else:
        print("Grad-CAM image was not saved!")

    return gradcam_image_path

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        print(f"Request files: {request.files}")  
        if 'file' not in request.files:
            print("No file part in request")  
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            print("No selected file")  
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            print(f"Saving file to {filepath}")
            file.save(filepath)

            if os.path.exists(filepath):
                print(filepath)
                print(f"File saved successfully: {filepath}") 
            else:
                print("File was not saved!")  

            gradcam_image_path = process_image(filepath)
            return render_template('result.html', filepath=filepath, gradcam_image_path=url_for('gradcam_images', filename=os.path.basename(gradcam_image_path)))

    return render_template('upload.html')

@app.route('/predict', methods=['GET'])
def predict():
    filepath = request.args.get('filepath')
    gradcam_image_path = process_image(filepath)
    ensemble = get_ensemble_model()
    predicted_disease = predict_disease(gradcam_image_path, ensemble)

    if predicted_disease in disease_info:
        info = disease_info[predicted_disease]
    else:
        info = {'Overview': 'N/A', 'Symptoms': [], 'Treatments': [], 'Precautionary Methods': []}

    response = {
        'predicted_disease': predicted_disease,
        'overview': info['Overview'],
        'symptoms': ', '.join(info['Symptoms']),
        'treatments': ', '.join(info['Treatments']),
        'precautionary_methods': ', '.join(info['Precautionary Methods'])
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
