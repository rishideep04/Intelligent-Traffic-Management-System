from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import threading
import vehicle_density1  # Import your YOLO vehicle density script
from werkzeug.utils import secure_filename
import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase Admin SDK (only once)
if not firebase_admin._apps:
    cred = credentials.Certificate("cloudproject-97507-firebase-adminsdk-fbsvc-96a5a3ca05.json")  # Replace with your service account key path
    firebase_admin.initialize_app(cred)

db = firestore.client()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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
            daemon=True  # Ensure threads exit when the main app exits
        ).start()

    return jsonify({"message": "Videos uploaded and processing started.", "file_paths": file_paths}), 200

def process_video_with_update(video_path, filename):
    """Process video and update density data."""
    try:
        # Call the YOLO-based vehicle density script with required arguments
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
            # Store density data in Firestore
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
    """Fetch density data from Firestore in chronological order (oldest to newest)."""
    try:
        print("🔍 Fetching density data from Firestore in chronological order...")
        
        # Query Firestore for density updates, sorted by timestamp in ascending order
        density_updates = (
            db.collection("density_updates")
            .order_by("timestamp", direction=firestore.Query.ASCENDING)
            .get()
        )
        
        # Initialize data structure to maintain chronological order
        video_data = {}

        for doc in density_updates:
            data = doc.to_dict()
            print(f"🔹 Processing document with timestamp: {data.get('timestamp')}")
            
            video_name = data.get("video")
            density = data.get("density")
            segment = data.get("segment")
            timestamp = data.get("timestamp")

            if None in [video_name, density, segment, timestamp]:
                print(f"❌ Skipping document with missing fields: {data}")
                continue

            if video_name not in video_data:
                video_data[video_name] = []
            
            # Store data with timestamp for proper ordering
            video_data[video_name].append({
                "density": density,
                "segment": segment,
                "timestamp": timestamp
            })

        # Prepare response data in chronological order
        response_data = {}
        for video_name, entries in video_data.items():
            # Sort entries by timestamp (though Firestore already returned them ordered)
            sorted_entries = sorted(entries, key=lambda x: x["timestamp"])
            response_data[video_name] = [entry["density"] for entry in sorted_entries]
            print(f"📊 Video '{video_name}' data (chronological): {response_data[video_name]}")

        print(f"📤 Sending response data for {len(response_data)} videos")
        return jsonify(response_data), 200
        
    except Exception as e:
        print(f"❌ Error fetching density data: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
