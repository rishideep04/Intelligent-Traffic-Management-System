from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import threading
import vehicle_density1
from werkzeug.utils import secure_filename
import firebase_admin
from firebase_admin import credentials, firestore
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from torch import nn

# Initialize Firebase Admin SDK
if not firebase_admin._apps:
    cred = credentials.Certificate("cloudproject-97507-firebase-adminsdk-fbsvc-96a5a3ca05.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# LSTM Model Definition
class Seq2SeqModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=300, num_layers=1):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, encoder_input, decoder_input):
        _, (hidden, cell) = self.encoder(encoder_input)
        decoder_output, _ = self.decoder(decoder_input, (hidden, cell))
        output = self.fc(decoder_output)
        return output

# Load LSTM model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Seq2SeqModel().to(device)
model.load_state_dict(torch.load("best_model_Seq2Seq.pt", map_location=device))
model.eval()

# Global variables to store recent data
recent_data = {
    "video1": [],
    "video2": []
}
LOOKBACK = 10 # Should match what you used in training

def preprocess_data(values):
    """Simplified preprocessing without time interpolation"""
    series = pd.Series(values)
    
    # Basic interpolation (not time-based)
    series = series.interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')
    
    # Simple smoothing
    if len(series) > 3:
        series = series.rolling(window=3, center=True, min_periods=1).mean()
    
    return series.values.astype('float32')

def prepare_input(data, LOOKBACK):
    """Prepare input tensor for model prediction"""
    if len(data) < LOOKBACK:
        # Pad with zeros if not enough data
        padded = np.zeros(LOOKBACK)
        padded[-len(data):] = data
        data = padded
    else:
        data = data[-LOOKBACK:]
    
    # Convert to tensor
    input_tensor = torch.FloatTensor(data).unsqueeze(0).unsqueeze(-1).to(device)
    decoder_input = input_tensor[:, -1:, :].repeat(1, LOOKBACK, 1)
    return input_tensor, decoder_input

@app.route('/')
def main_file():
    return render_template('main_page.html')

@app.route('/main_page.html')
def main_page():
    return render_template('main_page.html')

@app.route('/layout.html')
def layout_page():
    return render_template('layout.html')

@app.route('/dashboard.html')
def dashboard_page():
    return render_template('dashboard.html')

@app.route('/index.html')
def index_page():
    return render_template('index.html')

@app.route('/upload2.html')
def upload_file():
    return render_template('upload2.html')

@app.route('/simulation.html')
def simulation_page():
    return render_template('simulation.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle multiple video file uploads and start processing them in separate threads."""
    if 'file1' not in request.files or 'file2' not in request.files:
        return jsonify({"error": "Both video files are required."}), 400
    
    file1 = request.files['file1']
    file2 = request.files['file2']
    
    if file1.filename == '' or file2.filename == '':
        return jsonify({"error": "Both files must be selected."}), 400

    # Validate file extensions
    allowed_extensions = {'mp4'}
    if ('.' not in file1.filename or
        '.' not in file2.filename or
        file1.filename.rsplit('.', 1)[1].lower() not in allowed_extensions or
        file2.filename.rsplit('.', 1)[1].lower() not in allowed_extensions):
        return jsonify({"error": "Invalid file type. Only .mp4 files are allowed."}), 400

    file_paths = {}

    for file in [file1, file2]:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        file_paths[filename] = file_path
        print(f"📂 Saved file: {filename} at {file_path}")

    # Start processing both videos in separate threads
    for filename, file_path in file_paths.items():
        threading.Thread(
            target=process_video_with_update,
            args=(file_path, filename),
            daemon=True
        ).start()

    return jsonify({"message": "Videos uploaded and processing started.", "file_paths": file_paths}), 200

def process_video_with_update(video_path, filename):
    """Process video and update density data."""
    try:
        vehicle_density1.process_video(video_path, filename)
    except Exception as e:
        print(f"❌ Error processing video {filename}: {e}")

@app.route('/update_density', methods=['POST'])
def update_density():
    """Endpoint to receive density updates from the YOLO script."""
    try:
        data = request.json
        video_name = data.get('video')
        density = data.get('density')
        segment = data.get('segment')

        if video_name and density is not None and segment is not None:
            # Update recent data
            if "traffic_video.mp4" in video_name:
                recent_data["video1"].append(density)
                recent_data["video1"] = recent_data["video1"][-100:]  # Keep last 100 values
            elif "traffic_video1" in video_name:
                recent_data["video2"].append(density)
                recent_data["video2"] = recent_data["video2"][-100:]  # Keep last 100 values
            
            # Store in Firestore
            doc_ref = db.collection("density_updates").document()
            doc_ref.set({
                "density": density,
                "video": video_name,
                "segment": segment,
                "timestamp": firestore.SERVER_TIMESTAMP,
            })
            return jsonify({'message': 'Density updated'}), 200
        else:
            return jsonify({'error': 'Invalid data'}), 400
    except Exception as e:
        print(f"❌ Error processing request: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500

@app.route('/visualization2.html')
def visualization():
    return render_template('visualization2.html')

@app.route('/teamMembers.html')
def team_file():
    return render_template('teamMembers.html')

@app.route('/density_data', methods=['GET'])
def get_density_data():
    """Fetch density data from Firestore in chronological order."""
    try:
        print("🔍 Fetching density data from Firestore...")
        
        # Query Firestore for density updates, sorted by timestamp
        density_updates = (
            db.collection("density_updates")
            .order_by("timestamp", direction=firestore.Query.ASCENDING)
            .get()
        )
        
        # Initialize data structure
        video_data = {
            "traffic_video.mp4": [],
            "traffic_video1_-_Made_with_Clipchamp.mp4": []
        }

        for doc in density_updates:
            data = doc.to_dict()
            video_name = data.get("video")
            density = data.get("density")
            
            if "traffic_video.mp4" in video_name:
                video_data["traffic_video.mp4"].append(density)
            elif "traffic_video1" in video_name:
                video_data["traffic_video1_-_Made_with_Clipchamp.mp4"].append(density)

        print(f"📤 Sending response data")
        return jsonify(video_data), 200
        
    except Exception as e:
        print(f"❌ Error fetching density data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Validate input
        if not data or 'road1_data' not in data or 'road2_data' not in data:
            return jsonify({"error": "Missing road data in request"}), 400
        
        road1_data = data['road1_data']
        road2_data = data['road2_data']
        
        # Convert to numpy arrays
        try:
            road1_processed = preprocess_data(road1_data)
            road2_processed = preprocess_data(road2_data)
        except Exception as e:
            return jsonify({"error": f"Data processing error: {str(e)}"}), 400
        
        # Prepare model inputs
        def prepare_input(data):
            if len(data) < LOOKBACK:
                padded = np.zeros(LOOKBACK)
                padded[-len(data):] = data
                data = padded
            else:
                data = data[-LOOKBACK:]
            return torch.FloatTensor(data).unsqueeze(0).unsqueeze(-1).to(device)
        
        road1_input = prepare_input(road1_processed)
        road2_input = prepare_input(road2_processed)
        
        # Generate predictions
        predictions = []
        current_r1 = road1_input
        current_r2 = road2_input
        
        with torch.no_grad():
            for step in range(10):  # Predict next 10 steps
                # Road1 prediction
                dec1 = current_r1[:, -1:, :].repeat(1, LOOKBACK, 1)
                pred1 = model(current_r1, dec1)[:, -1, :].item()
                
                # Road2 prediction
                dec2 = current_r2[:, -1:, :].repeat(1, LOOKBACK, 1)
                pred2 = model(current_r2, dec2)[:, -1, :].item()
                
                predictions.append({
                    "step": step + 1,
                    "side": "road1" if pred1 > pred2 else "road2",
                    "confidence": max(pred1, pred2),
                    "road1_pred": pred1,
                    "road2_pred": pred2
                })
                
                # Update inputs for next step
                current_r1 = torch.cat([
                    current_r1[:, 1:, :], 
                    torch.tensor([[pred1]]).unsqueeze(0).to(device)
                ], dim=1)
                current_r2 = torch.cat([
                    current_r2[:, 1:, :], 
                    torch.tensor([[pred2]]).unsqueeze(0).to(device)
                ], dim=1)
        
        return jsonify({"predictions": predictions})
        
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
