import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firestore
def init_firestore():
    try:
        # Load your service account key
        cred = credentials.Certificate("cloudproject-97507-firebase-adminsdk-fbsvc-96a5a3ca05.json")
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("✅ Firestore initialized successfully!")
        return db
    except Exception as e:
        print(f"Error initializing Firestore: {e}")
        return None

# Save density data to Firestore
def save_density_to_firestore(db, video_name, segment, density):
    try:
        density_data = {
            "video_name": video_name,
            "segment": segment,
            "density": density
        }
        db.collection("traffic_density").add(density_data)
        print(f"✅ Density data saved to Firestore: {density_data}")
    except Exception as e:
        print(f"Error saving data to Firestore: {e}")
