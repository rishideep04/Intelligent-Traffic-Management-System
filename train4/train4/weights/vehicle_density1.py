from ultralytics import YOLO
import cv2
import time
import threading
import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase Admin SDK
cred = credentials.Certificate("cloudproject-97507-firebase-adminsdk-fbsvc-96a5a3ca05.json")  # Replace with your service account key path
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load the YOLO model
model = YOLO("best.pt")

def send_density_update(density, video_name, segment):
    try:
        # Store density data in Firestore
        doc_ref = db.collection("density_updates").document()
        doc_ref.set({
            "density": density,
            "video": video_name,
            "segment": segment,  # Include segment information
            "timestamp": firestore.SERVER_TIMESTAMP,
        })
        print(f"📤 Sent density update for {video_name}, Segment {segment} to Firestore: {density}")
    except Exception as e:
        print(f"Error sending density update for {video_name}: {e}")

def process_video(video_path, video_name):
    cap = cv2.VideoCapture(video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps if fps > 0 else 1  # Avoid division by zero
    segment_duration = 120  # 2 minutes per segment
    num_segments = max(1, int(video_duration // segment_duration))

    AVG_VEHICLE_LENGTH = 4.5
    segment = 1

    while cap.isOpened() and segment <= num_segments:
        vehicle_counts = []
        road_lengths = []

        start_time = time.time()

        while time.time() - start_time < segment_duration and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  

            results = model(frame)
            vehicle_boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy()
            
            vehicle_count = len(vehicle_boxes)
            vehicle_counts.append(vehicle_count)

            # Estimate road length
            if len(vehicle_boxes) > 1:
                x_coords = [box[0] for box in vehicle_boxes]
                min_x, max_x = min(x_coords), max(x_coords)
                road_length = ((max_x - min_x) / frame.shape[1]) * AVG_VEHICLE_LENGTH * len(vehicle_boxes)
            else:
                road_length = AVG_VEHICLE_LENGTH * max(vehicle_count, 1)

            road_lengths.append(road_length)
            
            for i, box in enumerate(vehicle_boxes):
                x1, y1, x2, y2 = map(int, box)
                confidence = confidences[i]
            
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Vehicle {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.putText(frame, f'Segment {segment}: Vehicles: {vehicle_count}', 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)    

            # Show the frame inside the loop
            cv2.imshow(f"Vehicle Detection - {video_name}", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break  

        # Ensure valid density calculation
        if vehicle_counts and road_lengths:
            avg_vehicles = int(sum(vehicle_counts) / len(vehicle_counts))  
            avg_road_length = sum(road_lengths) / len(road_lengths)

            if avg_road_length > 0:
                density = avg_vehicles / avg_road_length  

                print(f"🚗 Segment {segment} ({video_name}): Density = {density:.4f} vehicles/m")  

                # Send update to Firestore
                send_density_update(density, video_name, segment)  # Pass segment to the function
                print(f"📤 Sent density update for {video_name}, Segment {segment}: {density}")  

        segment += 1

    cap.release()
    cv2.destroyAllWindows()

# Threaded function to process two videos concurrently
def process_videos_concurrently(video_paths):
    threads = []

    # Create a thread for each video
    for i, video_path in enumerate(video_paths):
        video_name = f"Video_{i + 1}"
        thread = threading.Thread(target=process_video, args=(video_path, video_name))
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

# Example usage with two video paths
video_paths = ["video1.mp4", "video2.mp4"]
process_videos_concurrently(video_paths)
