import mediapipe as mp
import cv2
import numpy as np
import joblib
import time
from datetime import datetime
import csv
import os

class FaceMeshDetector:
    def __init__(self, session_id):
        # Mediapipe initialization
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(color=(128,0,128), thickness=2, circle_radius=1)
        
        # Load the pre-trained model
        with open('./lm_rf_model_04.p', 'rb') as model_file:
            rf_classifier = joblib.load(model_file)
            self.clf = rf_classifier['model']
            
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # inisialisasi sesi
        self.initialize_session(session_id)
    
    def initialize_session(self, session_id):
        """Initialize session-specific folders and logging files."""
        self.session_id = session_id
        
        # membuat directory sesi perekaman
        self.session_folder = os.path.join('sessions', self.session_id)
        self.images_folder = os.path.join(self.session_folder, 'images')
        
        # membuat directory
        os.makedirs(self.session_folder, exist_ok=True)
        os.makedirs(self.images_folder, exist_ok=True)
        
        # Initialize log file
        self.log_file = os.path.join(self.session_folder, f'detection_log_{self.session_id}.csv')
        self.initialize_logging()
    
    def initialize_logging(self):
        """Initialize the CSV log file with headers."""
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Timestamp', 
                'Face_Detected', 
                'Class', 
                'Computational_Time',
                'Image_Path'
            ])
    
    def save_frame(self, frame, prediction):
        """Save the current frame to the images folder."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        image_filename = f'frame_{timestamp}_{prediction if prediction else "no_face"}.jpg'
        image_path = os.path.join(self.images_folder, image_filename)
        cv2.imwrite(image_path, frame)
        return os.path.join('images', image_filename)
    
    def log_detection(self, face_detected, prediction, comp_time, image_path):
        """Log detection results to CSV file."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                face_detected,
                prediction if prediction else 'None',
                f"{comp_time:.4f}",
                image_path
            ])

    def process_frame(self, frame):
        start_time = time.time()
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        comp_time = time.time() - start_time
        
        if results.multi_face_landmarks:
            image, bbox, prediction = self.analyze_face(image, results.multi_face_landmarks[0])
            image_path = self.save_frame(image, prediction)
            self.log_detection(1, prediction, comp_time, image_path)
            return image, bbox, prediction, comp_time
        
        image_path = self.save_frame(image, None)
        self.log_detection(0, None, comp_time, image_path)
        return image, None, None, comp_time

    def analyze_face(self, image, face_landmarks):
        """Analyze detected face and draw annotations."""
        img_h, img_w, _ = image.shape

        coords = [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
        flattened_landmarks = np.array(coords).flatten().reshape(1, -1)
        
        prediction = self.clf.predict(flattened_landmarks)[0]
        
        x_coordinates = [landmark.x * img_w for landmark in face_landmarks.landmark]
        y_coordinates = [landmark.y * img_h for landmark in face_landmarks.landmark]
        
        x_min = int(min(x_coordinates))
        x_max = int(max(x_coordinates))
        y_min = int(min(y_coordinates))
        y_max = int(max(y_coordinates))
        
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(img_w, x_max + padding)
        y_max = min(img_h, y_max + padding)
        
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        text = f"Prediction: {prediction}"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(image, 
                    (x_min - 5, y_min - text_h - 15),
                    (x_min + text_w + 5, y_min - 5),
                    (0, 255, 0), -1)
        cv2.putText(image, text, (x_min, y_min - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        return image, (x_min, y_min, x_max, y_max), prediction
