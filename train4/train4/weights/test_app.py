import unittest
import os
import tempfile
from flask import Flask, jsonify
from flask_testing import TestCase
from app2 import app  # Import your Flask app
import io

class TestApp(TestCase):

    def create_app(self):
        app.config['TESTING'] = True
        app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
        return app

    def setUp(self):
        # Create a test client
        self.client = app.test_client()
        self.client.testing = True

    def tearDown(self):
        # Clean up after each test
        pass

    def test_main_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Traffic Light', response.data)

    def test_upload_page(self):
        response = self.client.get('/upload2.html')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Upload', response.data)

    def test_upload_video(self):
        # Create dummy files for testing
        file1 = (io.BytesIO(b"dummy file content"), 'test1.mp4')
        file2 = (io.BytesIO(b"dummy file content"), 'test2.mp4')

        response = self.client.post('/upload', data={
            'file1': file1,
            'file2': file2
        }, content_type='multipart/form-data')

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Videos uploaded and processing started', response.data)
    
    def test_upload_invalid_file(self):
        # Test uploading a non-video file
        file1 = (io.BytesIO(b"dummy file content"), 'test1.txt')
        file2 = (io.BytesIO(b"dummy file content"), 'test2.txt')

        response = self.client.post('/upload', data={
            'file1': file1,
            'file2': file2
        }, content_type='multipart/form-data')

        self.assertEqual(response.status_code, 400)
        self.assertIn(b'Invalid file type', response.data)
    def test_upload_missing_files(self):
        # Test uploading without files
        response = self.client.post('/upload', data={}, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)
        self.assertIn(b'Both video files are required', response.data)
    
    def test_upload_large_file(self):
        # Test uploading a large file
        large_file = (io.BytesIO(b"0" * 100 * 1024 * 1024), 'large_file.mp4')  # 100 MB file
        response = self.client.post('/upload', data={
            'file1': large_file,
            'file2': large_file
        }, content_type='multipart/form-data')

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Videos uploaded and processing started', response.data)

    def test_update_density(self):
        # Test the /update_density endpoint
        data = {
            "video": "test_video.mp4",
            "density": 10,
            "segment": 1
        }
        response = self.client.post('/update_density', json=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Density updated', response.data)

    def test_get_density_data(self):
        # Test the /density_data endpoint
        response = self.client.get('/density_data')
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json, dict)

    def test_simulation_page(self):
        response = self.client.get('/simulation.html')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Simulation', response.data)

    def test_visualization_page(self):
        response = self.client.get('/visualization2.html')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Visualization', response.data)

    def test_team_members_page(self):
        response = self.client.get('/teamMembers.html')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Meet Our Team!', response.data)

if __name__ == '__main__':
    unittest.main()
