import os
import io
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__, template_folder='.')

# Construct the path to the model file
model_path = os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'models', 'cifar10_model.h5')
model = tf.keras.models.load_model(model_path)

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    try:
        # IMPORTANT: Seek to the beginning of the file
        file.seek(0)
        
        # Open and preprocess the image
        img = Image.open(file).convert('RGB')
        img = img.resize((32, 32))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Make prediction
        predictions = model.predict(img_array)
        
        predicted_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_index]
        confidence = float(predictions[0][predicted_index])
        
        all_predictions = {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}
        
        return jsonify({
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_predictions': all_predictions
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)