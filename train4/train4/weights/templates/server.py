import asyncio
import websockets
import cv2
import numpy as np

async def send_frames(websocket):
    cap0 = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # First camera
    cap1 = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Second camera

    if not cap0.isOpened() or not cap1.isOpened():
        print("Failed to access one or both cameras")
        return
    
    try:
        while True:
            ret0, frame0 = cap0.read()
            ret1, frame1 = cap1.read()

            if not ret0 or not ret1:
                print("Failed to capture frame from one or both cameras")
                break
            
            # Encode frames to JPEG
            _, buffer0 = cv2.imencode('.jpg', frame0)
            _, buffer1 = cv2.imencode('.jpg', frame1)

            # Send both frames as binary data
            await websocket.send(buffer0.tobytes())
            await websocket.send(buffer1.tobytes())

            await asyncio.sleep(0.033)  # ~30 FPS

    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")

    finally:
        cap0.release()
        cap1.release()

async def main():
    async with websockets.serve(send_frames, "localhost", 8765):
        print("🚀 WebSocket server running at ws://localhost:8765")
        await asyncio.Future()  # Keep server running

if __name__ == "__main__":
    asyncio.run(main())
