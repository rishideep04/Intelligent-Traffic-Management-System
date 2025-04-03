import unittest
import asyncio
import websockets
import cv2
import numpy as np
from unittest.mock import MagicMock, patch
from server import send_frames  # Import your server function

class TestWebSocketServer(unittest.TestCase):
    def setUp(self):
        # Set up the event loop for testing
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        # Clean up the event loop
        self.loop.close()

    async def connect_and_receive(self, uri):
        """Helper function to connect to the WebSocket server and receive frames."""
        async with websockets.connect(uri) as websocket:
            # Receive two frames (one from each camera)
            frame0 = await websocket.recv()
            frame1 = await websocket.recv()
            return frame0, frame1

    def test_send_frames(self):
        """Test that the server sends frames correctly."""
        async def test():
            # Mock cv2.VideoCapture
            mock_cap0 = MagicMock()
            mock_cap1 = MagicMock()

            # Mock frames
            mock_frame0 = np.zeros((480, 640, 3), dtype=np.uint8)  # Black image
            mock_frame1 = np.ones((480, 640, 3), dtype=np.uint8) * 255  # White image

            # Mock read() method
            mock_cap0.read.return_value = (True, mock_frame0)
            mock_cap1.read.return_value = (True, mock_frame1)

            # Mock isOpened() method
            mock_cap0.isOpened.return_value = True
            mock_cap1.isOpened.return_value = True

            # Patch cv2.VideoCapture
            with patch('cv2.VideoCapture', side_effect=[mock_cap0, mock_cap1]):
                # Start the WebSocket server
                server = await websockets.serve(send_frames, "localhost", 8765)

                try:
                    # Connect to the server and receive frames
                    frame0, frame1 = await self.connect_and_receive("ws://localhost:8765")

                    # Check that the received data is valid
                    self.assertIsInstance(frame0, bytes)
                    self.assertIsInstance(frame1, bytes)

                    # Decode the frames to ensure they are valid images
                    img0 = cv2.imdecode(np.frombuffer(frame0, dtype=np.uint8), cv2.IMREAD_COLOR)
                    img1 = cv2.imdecode(np.frombuffer(frame1, dtype=np.uint8), cv2.IMREAD_COLOR)

                    self.assertIsNotNone(img0)
                    self.assertIsNotNone(img1)

                finally:
                    # Stop the server
                    server.close()
                    await server.wait_closed()

        # Run the test in the event loop
        self.loop.run_until_complete(test())

if __name__ == "__main__":
    unittest.main()
