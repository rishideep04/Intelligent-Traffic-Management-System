import unittest
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
from vehicle_density1 import process_video, send_density_update  # Import your functions
from firebase_admin import firestore  # Import firestore module
from colorama import Fore, Style, init  # Import colorama for colored output

# Initialize colorama
init()

class TestVehicleDensity(unittest.TestCase):
    def setUp(self):
        # Mock Firebase Firestore
        self.mock_db = MagicMock()
        self.mock_doc_ref = MagicMock()
        self.mock_db.collection.return_value.document.return_value = self.mock_doc_ref

        # Patch Firebase Firestore
        self.firestore_patch = patch('vehicle_density1.db', self.mock_db)
        self.firestore_patch.start()

    def tearDown(self):
        # Stop patching Firebase Firestore
        self.firestore_patch.stop()

    @patch('vehicle_density1.cv2.VideoCapture')
    @patch('vehicle_density1.model')
    def test_process_video(self, mock_model, mock_video_capture):
        """Test the process_video function."""
        # Mock video capture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = [30, 1800]  # FPS = 30, total_frames = 1800
        mock_cap.read.side_effect = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),  # Frame 1
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),  # Frame 2
            (False, None)  # End of video
        ]
        mock_video_capture.return_value = mock_cap

        # Mock YOLO model results
        mock_results = MagicMock()
        mock_results[0].boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[0, 0, 100, 100]])
        mock_results[0].boxes.conf.cpu.return_value.numpy.return_value = np.array([0.9])
        mock_results[0].boxes.cls.cpu.return_value.numpy.return_value = np.array([0])
        mock_model.return_value = mock_results

        # Call the function
        process_video("dummy_video.mp4", "Test Video")

        # Assertions
        mock_cap.release.assert_called_once()
        mock_model.assert_called()  # Ensure the model was called
        self.mock_doc_ref.set.assert_called()  # Ensure Firestore was updated

    @patch('vehicle_density1.db')
    def test_send_density_update(self, mock_db):
        """Test the send_density_update function."""
        # Call the function
        send_density_update(0.5, "Test Video", 1)

        # Assertions
        mock_db.collection.return_value.document.return_value.set.assert_called_once_with({
            "density": 0.5,
            "video": "Test Video",
            "segment": 1,
            "timestamp": firestore.SERVER_TIMESTAMP,  # Use the imported firestore module
        })

if __name__ == "__main__":
    # Run tests with colored output
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestVehicleDensity)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print colored summary
    if result.wasSuccessful():
        print(Fore.GREEN + "ALL TESTS PASSED!" + Style.RESET_ALL)
    else:
        print(Fore.RED + "SOME TESTS FAILED!" + Style.RESET_ALL)
