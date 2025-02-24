from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
from werkzeug.utils import secure_filename
import threading
import vehicle_density  # Import your YOLO vehicle density script

density_data = []  # Store density values to visualize later

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def send_density_update(density):
    global density_data
    density_data.append(density)

@app.route('/')
def upload_file():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Start processing in a separate thread
        threading.Thread(target=vehicle_density.process_video, args=(file_path,)).start()
        
        return redirect(url_for('visualization'))
@app.route('/update_density', methods=['POST'])
def update_density():
    global density_data
    data = request.get_json()
    density = data.get('density')
    if density is not None:
        density_data.append(density)
        print(f"🔹 New density data received: {density_data}")  # Debugging print statement
    return jsonify({"status": "success", "data": density_data})



@app.route('/visualization')
def visualization():
    return render_template('visualization.html')

@app.route('/density_data')
def get_density_data():
    return jsonify(density_data)

if __name__ == '__main__':
    app.run(debug=True)