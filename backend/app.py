import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, jsonify
import torch
import cv2
import numpy as np
from PIL import Image
import io
import base64
from torchvision import transforms
from model.mobilenet_model import get_mobilenet_model

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../model/best_qr_model.pth')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_mobilenet_model().to(device)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Model loaded successfully!")
else:
    print(f"Warning: Model not found at {MODEL_PATH}")

# Image preprocessing - MUST match training normalization!
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read and preprocess image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Transform image
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
            
        # Get result
        class_names = ['Original', 'Fake']
        prediction = class_names[predicted.item()]
        confidence_score = confidence.item() * 100
        
        # Convert image to base64 for display
        buffered = io.BytesIO(image_bytes)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({
            'prediction': prediction,
            'confidence': f'{confidence_score:.2f}',
            'image': f'data:image/jpeg;base64,{img_str}',
            'probabilities': {
                'Original': f'{probs[0][0].item() * 100:.2f}',
                'Fake': f'{probs[0][1].item() * 100:.2f}'
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
