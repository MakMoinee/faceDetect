from flask import Flask, render_template_string, request, jsonify
import torch
from PIL import Image
import io
import base64
import numpy as np
import cv2
import os
import shutil
import yaml
import json
import subprocess
import time
import random

from flask_cors import CORS

import pathlib

# Temporarily set PosixPath to WindowsPath to resolve potential path issues on Windows
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

app = Flask(__name__)
CORS(app)

# Define base directory for cleaner absolute path management
BASE_DIR = "D:\\work\\faceDetect"

# Define dataset paths
DATASET_ROOT = os.path.join(BASE_DIR, 'dataset')
IMAGES_TRAIN_DIR = os.path.join(DATASET_ROOT, 'images', 'train')
LABELS_TRAIN_DIR = os.path.join(DATASET_ROOT, 'labels', 'train')

IMAGES_VAL_DIR = os.path.join(DATASET_ROOT, 'images', 'val')
LABELS_VAL_DIR = os.path.join(DATASET_ROOT, 'labels', 'val')

VAL_RATIO = 0.3

# File to store name-to-ID mapping and global counters
CLASS_MAPPING_FILE = os.path.join(BASE_DIR, 'class_mapping.json')
DATA_YAML_PATH = os.path.join(BASE_DIR, 'data.yaml')

# Default YOLOv5 pre-trained model for initial detection
DEFAULT_YOLOV5_MODEL = os.path.join(BASE_DIR, 'yolov5s.pt')

# --- CORRECTED PATH DEFINITIONS FOR YOLOv5 OUTPUT ---
YOLOV5_PROJECT_ARG_NAME = 'custom_yolov5_training_project'
YOLOV5_RUN_ARG_NAME = 'face_recognizer_run'

YOLOV5_RUNS_ROOT_DIR = os.path.join(BASE_DIR, 'runs')

# The specific directory where this run's results will be stored
CURRENT_RUN_DIR = os.path.join(YOLOV5_PROJECT_ARG_NAME, YOLOV5_RUN_ARG_NAME)

TRAINED_MODEL_WEIGHTS_DIR = os.path.join(
    CURRENT_RUN_DIR, 'weights'
)
TRAINED_MODEL_FULL_PATH = os.path.join(TRAINED_MODEL_WEIGHTS_DIR, 'best.pt')


# Global variables for loaded models and current model type
inference_model = None
training_model = None
current_loaded_model_type = 'unknown' # 'default' or 'trained'

# Global counters and mapping
class_name_to_id = {}
next_class_id = 0
image_count = 0
finished_class_ids = set()  # Track class IDs that are marked as finished

# --- Helper Functions ---

def save_class_mapping():
    """Saves the current class name to ID mapping and counters to a JSON file."""
    with open(CLASS_MAPPING_FILE, 'w') as f:
        json.dump({
            'class_name_to_id': class_name_to_id,
            'next_class_id': next_class_id,
            'image_count': image_count,
            'finished_class_ids': list(finished_class_ids)  # Convert set to list for JSON serialization
        }, f, indent=4)

def update_data_yaml():
    """Updates the data.yaml file with current class names and dataset paths."""
    global class_name_to_id
    
    # Sort names by their assigned ID to ensure consistent order in data.yaml
    # This creates a list of (name, id) tuples, then convert to a dict
    sorted_class_items = sorted(class_name_to_id.items(), key=lambda item: item[1])
    names_dict = {idx: name for name, idx in sorted_class_items}

    data_config = {
        'train': str(pathlib.Path(IMAGES_TRAIN_DIR).resolve()),
        'val': str(pathlib.Path(IMAGES_VAL_DIR).resolve()),
        'nc': len(names_dict),
        'names': names_dict # NOW A DICTIONARY!
    }
    with open(DATA_YAML_PATH, 'w') as f:
        yaml.dump(data_config, f, sort_keys=False) # sort_keys=False preserves insertion order in YAML for Python 3.7+

def load_detection_model(model_to_load_path=None):
    """
    Loads the YOLOv5 model for inference.
    Prioritizes the custom trained model if it exists, otherwise falls back to default.
    """
    global inference_model, training_model, current_loaded_model_type

    final_model_path = None
    if model_to_load_path is None:
        if os.path.exists(TRAINED_MODEL_FULL_PATH):
            final_model_path = TRAINED_MODEL_FULL_PATH
            current_loaded_model_type = 'trained'
            print(f"Loading trained model: {final_model_path}")
        else:
            final_model_path = DEFAULT_YOLOV5_MODEL
            current_loaded_model_type = 'default'
            print(f"Trained model not found. Loading default model: {final_model_path}")
    else:
        final_model_path = model_to_load_path
        # Determine type based on provided path
        if final_model_path == DEFAULT_YOLOV5_MODEL:
            current_loaded_model_type = 'default'
        elif final_model_path == TRAINED_MODEL_FULL_PATH:
            current_loaded_model_type = 'trained'
        else:
            current_loaded_model_type = 'custom_specified' # Should not happen often
        print(f"Loading specified model: {final_model_path}")

    inference_model = torch.hub.load('ultralytics/yolov5', 'custom', path=final_model_path, force_reload=True)
    
    # Set confidence threshold based on model type
    if current_loaded_model_type == 'default':
        inference_model.conf = 0.2 # Lower confidence for 'person' detection during data collection
        print(f"Set inference model confidence to {inference_model.conf} for initial 'person' detection.")
    else: # For a trained custom model
        inference_model.conf = 0.5 # Default confidence for trained model, can be adjusted for production
        print(f"Set inference model confidence to {inference_model.conf} for trained custom model.")

    try:
        # Load without autoshape for compatibility if using model.train() directly,
        # but subprocess handles the training in this setup.
        training_model = torch.hub.load('ultralytics/yolov5', 'custom', path=final_model_path, autoshape=False, force_reload=True)
    except TypeError as e:
        print(f"Warning: Could not load training model with autoshape=False directly: {e}")
        print("Falling back to using inference model for 'training_model' placeholder.")
        training_model = inference_model

def initialize_app_data_and_models():
    """Initializes dataset directories, loads saved data, and loads the appropriate model."""
    global class_name_to_id, next_class_id, image_count, finished_class_ids

    os.makedirs(IMAGES_TRAIN_DIR, exist_ok=True)
    os.makedirs(LABELS_TRAIN_DIR, exist_ok=True)
    os.makedirs(IMAGES_VAL_DIR, exist_ok=True)
    os.makedirs(LABELS_VAL_DIR, exist_ok=True)

    if os.path.exists(CLASS_MAPPING_FILE):
        try:
            with open(CLASS_MAPPING_FILE, 'r') as f:
                data = json.load(f)
                class_name_to_id = data.get('class_name_to_id', {})
                next_class_id = data.get('next_class_id', 0)
                image_count = data.get('image_count', 0)
                finished_class_ids = set(data.get('finished_class_ids', []))  # Load finished class IDs
        except json.JSONDecodeError:
            print(f"Warning: {CLASS_MAPPING_FILE} is corrupted or empty. Starting with empty mapping.")
            class_name_to_id = {}
            next_class_id = 0
            image_count = 0
            finished_class_ids = set()
    
    update_data_yaml()
    load_detection_model()

initialize_app_data_and_models()

# --- Flask Routes ---

@app.route('/')
def index():
    return """
    <h1>Welcome!</h1>
    <p>Go to <a href="/stream">/stream</a> for live camera detection.</p>
    <p>Go to <a href="/data_collection_and_training">/data_collection_and_training</a> for collecting images and training your custom model.</p>
    """

@app.route('/stream')
def stream():
    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
    <title>Live Face Detection Stream</title>
    <style>
        video, canvas {
            position: absolute;
            left: 0;
            top: 0;
            width: 640px;
            height: 480px;
        }
        #container {
            position: relative;
            width: 640px;
            height: 480px;
        }
    </style>
</head>
<body>
    <h2>Click Start to open camera and start face detection</h2>
    <p>To detect a specific person, append `?person=NAME` to the URL (e.g., `/stream?person=JohnDoe`).</p>
    <button id="startBtn">Start Detect</button>
    <br>

    <div id="container">
        <video id="video" autoplay muted></video>
        <canvas id="canvas"></canvas>
    </div>

    <script>
        const startBtn = document.getElementById('startBtn');
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');

        let stream;
        let detecting = false;

        startBtn.onclick = async () => {
            if (detecting) return;

            try {
                if (navigator.permissions) {
                    navigator.permissions.query({name: 'camera'})
                        .then((permissionObj) => {
                            if (permissionObj.state === 'denied') {
                                alert('Camera access has been denied. Please enable it in browser settings.');
                                return;
                            }
                        })
                        .catch((error) => {
                            console.log('Got error :', error);
                        });
                }

                stream = await navigator.mediaDevices.getUserMedia({ video: true });
                video.srcObject = stream;
                detecting = true;
                detectFrame();
            } catch (e) {
                alert('Camera access denied or not available');
                console.error(e);
            }
        };

        async function detectFrame() {
            if (!detecting) return;

            const offscreen = document.createElement('canvas');
            offscreen.width = video.videoWidth;
            offscreen.height = video.videoHeight;
            const offctx = offscreen.getContext('2d');
            offctx.drawImage(video, 0, 0, offscreen.width, offscreen.height);

            const dataUrl = offscreen.toDataURL('image/jpeg', 0.7);
            const base64 = dataUrl.split(',')[1];

            // Get person name from URL query parameter
            const urlParams = new URLSearchParams(window.location.search);
            const personToDetect = urlParams.get('person');

            try {
                // Pass personToDetect to the backend
                const response = await fetch('/detect_frame' + (personToDetect ? `?person_name=${personToDetect}` : ''), {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: base64 })
                });
                const data = await response.json();

                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                ctx.clearRect(0, 0, canvas.width, canvas.height);

                ctx.lineWidth = 2;
                ctx.strokeStyle = 'red';
                ctx.font = '16px Arial';
                ctx.fillStyle = 'red';

                data.faces.forEach(face => {
                    const x = face.xmin;
                    const y = face.ymin;
                    const w = face.xmax - face.xmin;
                    const h = face.ymax - face.ymin;
                    ctx.strokeRect(x, y, w, h);
                    ctx.fillText(
                        `${face.name} (${(face.confidence*100).toFixed(1)}%)`,
                        x,
                        y > 20 ? y - 5 : y + 15
                    );
                });

            } catch (e) {
                console.error('Error detecting frame:', e);
            }

            setTimeout(detectFrame, 200);
        }
    </script>
</body>
</html>
    """)

@app.route('/data_collection_and_training')
def data_collection_and_training():
    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
    <title>Image Data Collection & Model Training</title>
    <style>
        video, canvas {
            position: absolute;
            left: 0;
            top: 0;
            width: 640px;
            height: 480px;
        }
        #container {
            position: relative;
            width: 640px;
            height: 480px;
            margin-bottom: 20px;
        }
        #controls {
            margin-top: 100px;
        }
    </style>
</head>
<body>
    <h2>Image Data Collection and Model Training</h2>
    <div id="controls">
        <button id="startCameraBtn">Start Camera</button>
        <input type="text" id="personName" placeholder="Enter name for face">
        <button id="captureImageBtn">Capture 30 Images</button>
        <p id="captureCount">Captured: 0 / 30</p>
        <button id="trainModelBtn">Train Model</button>
        <p id="message"></p>
        <h3>Registered Faces: <span id="registeredFaces"></span></h3>
    </div>
    
    <div id="container">
        <video id="video" autoplay muted></video>
        <canvas id="canvas"></canvas>
    </div>

    <script>
        const startCameraBtn = document.getElementById('startCameraBtn');
        const captureImageBtn = document.getElementById('captureImageBtn');
        const trainModelBtn = document.getElementById('trainModelBtn');
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const messageElement = document.getElementById('message');
        const personNameInput = document.getElementById('personName');
        const registeredFacesSpan = document.getElementById('registeredFaces');
        const captureCountElement = document.getElementById('captureCount');

        let stream;
        let cameraActive = false;
        const IMAGES_TO_CAPTURE = 30;
        let currentCapturedCount = 0;

        async function updateRegisteredFaces() {
            try {
                const response = await fetch('/get_registered_faces');
                const data = await response.json();
                if (data.success) {
                    const names = Object.keys(data.class_name_to_id).sort((a, b) => data.class_name_to_id[a] - data.class_name_to_id[b]);
                    registeredFacesSpan.innerText = names.join(', ') || 'None';
                } else {
                    registeredFacesSpan.innerText = 'Error fetching faces.';
                    console.error('Error fetching registered faces:', data.error);
                }
            } catch (e) {
                registeredFacesSpan.innerText = 'Error fetching faces.';
                console.error('Network error fetching registered faces:', e);
            }
        }

        updateRegisteredFaces();

        startCameraBtn.onclick = async () => {
            if (cameraActive) return;

            try {
                if (navigator.permissions) {
                    navigator.permissions.query({name: 'camera'})
                        .then((permissionObj) => {
                            if (permissionObj.state === 'denied') {
                                alert('Camera access has been denied. Please enable it in browser settings.');
                                return;
                            }
                        })
                        .catch((error) => {
                            console.log('Got error :', error);
                        });
                }

                stream = await navigator.mediaDevices.getUserMedia({ video: true });
                video.srcObject = stream;
                cameraActive = true;
            } catch (e) {
                alert('Camera access denied or not available');
                console.error(e);
            }
        };

        // Function to capture a single image
        async function captureSingleImage(personName) {
            const offscreen = document.createElement('canvas');
            offscreen.width = video.videoWidth;
            offscreen.height = video.videoHeight;
            const offctx = offscreen.getContext('2d');
            offctx.drawImage(video, 0, 0, offscreen.width, offscreen.height);

            const dataUrl = offscreen.toDataURL('image/jpeg', 0.7);
            const base64 = dataUrl.split(',')[1];

            try {
                const detectResponse = await fetch('/detect_frame', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: base64 })
                });
                const detectData = await detectResponse.json();

                // IMPORTANT: Ensure detections are present before proceeding
                if (!detectData.faces || detectData.faces.length === 0) {
                    return { success: false, error: "No faces (persons) detected in the frame. Please ensure your face is visible." };
                }

                // Send the captured image and detections to the Flask backend
                const captureResponse = await fetch('/capture_data', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: base64, detections: detectData.faces, name: personName })
                });
                const captureData = await captureResponse.json();

                if (captureData.success) {
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    ctx.lineWidth = 2;
                    ctx.strokeStyle = 'blue';
                    ctx.font = '16px Arial';
                    ctx.fillStyle = 'blue';
                    // Draw the detections on the client-side for immediate visual feedback
                    detectData.faces.forEach(face => {
                        const x = face.xmin;
                        const y = face.ymin;
                        const w = face.xmax - face.xmin;
                        const h = face.ymax - face.ymin;
                        ctx.strokeRect(x, y, w, h);
                        ctx.fillText(
                            `${face.name} (${(face.confidence*100).toFixed(1)}%)`,
                            x,
                            y > 20 ? y - 5 : y + 15
                        );
                    });
                    return { success: true, filename: captureData.filename, detections_count: detectData.faces.length };
                } else {
                    return { success: false, error: captureData.error };
                }
            } catch (e) {
                return { success: false, error: `Error during capture: ${e.message}` };
            }
        }

        captureImageBtn.onclick = async () => {
            if (!cameraActive) {
                alert('Please start the camera first.');
                return;
            }

            const personName = personNameInput.value.trim();
            if (!personName) {
                alert('Please enter a name for the face.');
                return;
            }

            captureImageBtn.disabled = true;
            trainModelBtn.disabled = true;
            messageElement.innerText = `Starting to capture ${IMAGES_TO_CAPTURE} images for ${personName}...`;
            currentCapturedCount = 0;
            captureCountElement.innerText = `Captured: ${currentCapturedCount} / ${IMAGES_TO_CAPTURE}`;


            for (let i = 0; i < IMAGES_TO_CAPTURE; i++) {
                messageElement.innerText = `Capturing image ${i + 1} of ${IMAGES_TO_CAPTURE} for ${personName}...`;
                const result = await captureSingleImage(personName);
                if (result.success) {
                    currentCapturedCount++;
                    captureCountElement.innerText = `Captured: ${currentCapturedCount} / ${IMAGES_TO_CAPTURE}`;
                    updateRegisteredFaces();
                } else {
                    messageElement.innerText = `Capture failed for image ${i + 1}: ${result.error}. Stopping capture.`;
                    break;
                }
                await new Promise(resolve => setTimeout(resolve, 500)); // 0.5 second delay between captures
            }

            if (currentCapturedCount === IMAGES_TO_CAPTURE) {
                messageElement.innerText = `Finished capturing ${IMAGES_TO_CAPTURE} images for ${personName}.`;
            } else {
                messageElement.innerText = `Stopped capturing. Captured ${currentCapturedCount} / ${IMAGES_TO_CAPTURE} images.`;
            }
            captureImageBtn.disabled = false;
            trainModelBtn.disabled = false;
        };

        trainModelBtn.onclick = async () => {
            messageElement.innerText = "Training model... This may take a while.";
            trainModelBtn.disabled = true;
            captureImageBtn.disabled = true;
            try {
                const response = await fetch('/train_model', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                });
                const data = await response.json();
                if (data.success) {
                    messageElement.innerText = `Training complete! Model saved to: ${data.model_path}. Reload /stream to use it.`;
                } else {
                    messageElement.innerText = `Training failed: ${data.error}`;
                }
            } catch (e) {
                messageElement.innerText = `Error during training request: ${e.message}`;
                console.error('Error during training request:', e);
            } finally {
                trainModelBtn.disabled = false;
                captureImageBtn.disabled = false; 
            }
        };
    </script>
</body>
</html>
    """)

@app.route('/detect_frame', methods=['POST'])
def detect_frame_api():
    """
    Receives a base64 encoded image, performs object detection using the currently loaded model,
    and returns detected faces with bounding box coordinates and names.
    
    Accepts an optional 'person_name' query parameter to filter detections for a specific person.
    """
    try:
        data = request.get_json()
        img_b64 = data['image']
        img_bytes = base64.b64decode(img_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        results = inference_model(img)
        detections_df = results.pandas().xyxy[0]

        faces = []
        model_class_names = inference_model.names
        
        # Get optional 'person_name' query parameter from request arguments
        target_person_name = request.args.get('person_name')
        
        # Validate target_person_name only if current_loaded_model_type is 'trained'
        # The 'default' model doesn't know about custom names, so no validation needed then.
        if target_person_name and current_loaded_model_type == 'trained':
            if target_person_name not in class_name_to_id:
                print(f"Warning: '{target_person_name}' is not a registered face. Ignoring specific person filter for trained model.")
                target_person_name = None # Revert to no filtering if name is invalid or not registered
            
        for _, row in detections_df.iterrows():
            detected_class_id = int(row['class'])
            detected_class_name = model_class_names[detected_class_id]
            
            # --- Filtering Logic ---
            
            # 1. If a specific person is requested (and it's a trained model), filter only that person.
            if target_person_name and current_loaded_model_type == 'trained':
                if detected_class_name == target_person_name:
                    faces.append({
                        "xmin": float(row['xmin']),
                        "ymin": float(row['ymin']),
                        "xmax": float(row['xmax']),
                        "ymax": float(row['ymax']),
                        "confidence": float(row['confidence']),
                        "class": detected_class_id,
                        "name": detected_class_name
                    })
                continue # Move to next detection if a target person is set

            # 2. If no specific person is requested OR it's the default model:
            if current_loaded_model_type == 'default':
                # For the default model, only consider "person" detections for display/capture
                if detected_class_name == "person":
                    faces.append({
                        "xmin": float(row['xmin']),
                        "ymin": float(row['ymin']),
                        "xmax": float(row['xmax']),
                        "ymax": float(row['ymax']),
                        "confidence": float(row['confidence']),
                        "class": detected_class_id,
                        "name": detected_class_name
                    })
            elif current_loaded_model_type == 'trained':
                # For a trained model, include all detections that correspond to registered names
                # (this already filters out unknown classes or noise if target_person_name is not set)
                if detected_class_name in class_name_to_id:
                    faces.append({
                        "xmin": float(row['xmin']),
                        "ymin": float(row['ymin']),
                        "xmax": float(row['xmax']),
                        "ymax": float(row['ymax']),
                        "confidence": float(row['confidence']),
                        "class": detected_class_id,
                        "name": detected_class_name
                    })
            # No 'else' needed here, as non-matching detections are skipped by default.

        # Debugging prints
        if not detections_df.empty:
            print(f"Raw detections from model ({current_loaded_model_type}): {detections_df.name.tolist()}")
        else:
            print(f"No raw detections from model ({current_loaded_model_type}).")

        print(f"Filtered relevant objects for display: {len(faces)}")
        if len(faces) == 0:
            # Added a more specific message if filtering occurred
            if target_person_name:
                print(f"No objects detected for target person '{target_person_name}' in this frame.")
            else:
                print("No relevant objects detected by the model in this frame.")
            print(f"Model: {current_loaded_model_type}, Classes: {model_class_names}")


        return jsonify({"faces": faces})
    except Exception as e:
        print(f"Error in detect_frame_api: {e}")
        return jsonify({"error": str(e), "faces": []}), 500

@app.route('/capture_data', methods=['POST'])
def capture_data():
    """
    Receives an image and initial detections, assigns a class ID based on provided name,
    saves the image, and creates a YOLOv5-formatted label file.
    Distributes images into train/val folders.
    
    If query parameter ?finished=true is present, marks the class ID as finished
    and prevents further image saving for that class ID.
    """
    global image_count
    global next_class_id
    global class_name_to_id
    global finished_class_ids

    try:
        # Check if finished=true query parameter is present
        finished_param = request.args.get('finished', '').lower() == 'true'
        
        data = request.get_json()
        img_b64 = data['image']
        detections = data['detections']
        person_name = data['name'].strip()

        if not person_name:
            return jsonify({"success": False, "error": "Person name is required."}), 400

        if person_name not in class_name_to_id:
            class_name_to_id[person_name] = next_class_id
            next_class_id += 1
            update_data_yaml()
            save_class_mapping()
            print(f"Registered new face: '{person_name}' with ID {class_name_to_id[person_name]}")

        person_class_id = class_name_to_id[person_name]
        
        # If finished=true, mark this class ID as finished and return
        if finished_param:
            finished_class_ids.add(person_class_id)
            save_class_mapping()
            print(f"Marked class ID {person_class_id} ({person_name}) as finished. No more images will be saved for this class.")
            return jsonify({"success": True, "message": f"Class ID {person_class_id} ({person_name}) marked as finished."})
        
        # Check if this class ID is already finished
        if person_class_id in finished_class_ids:
            return jsonify({"success": False, "error": f"Class ID {person_class_id} ({person_name}) is already marked as finished. No more images can be saved for this class."}), 400

        img_bytes = base64.b64decode(img_b64)
        img_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        image_count += 1
        img_filename = f"image_{image_count:05d}.jpg"
        label_filename = f"image_{image_count:05d}.txt"

        if random.random() < VAL_RATIO:
            target_image_dir = IMAGES_VAL_DIR
            target_label_dir = LABELS_VAL_DIR
            folder_type = "val"
        else:
            target_image_dir = IMAGES_TRAIN_DIR
            target_label_dir = LABELS_TRAIN_DIR
            folder_type = "train"

        os.makedirs(target_image_dir, exist_ok=True)
        os.makedirs(target_label_dir, exist_ok=True)

        img_path = os.path.join(target_image_dir, img_filename)
        label_path = os.path.join(target_label_dir, label_filename)

        img_pil.save(img_path)

        with open(label_path, 'w') as f:
            img_width, img_height = img_pil.size
            for det in detections:
                xmin = max(0, det['xmin'])
                ymin = max(0, det['ymin'])
                xmax = min(img_width, det['xmax'])
                ymax = min(img_height, det['ymax'])

                center_x = ((xmin + xmax) / 2) / img_width
                center_y = ((ymin + ymax) / 2) / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height

                f.write(f"{person_class_id} {center_x} {center_y} {width} {height}\n")

        save_class_mapping()
        print(f"Captured {img_filename} for {person_name} (ID: {person_class_id}) into {folder_type} set.")
        return jsonify({"success": True, "filename": img_filename, "folder": folder_type})
    except Exception as e:
        print(f"Error capturing data: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/get_registered_faces', methods=['GET'])
def get_registered_faces():
    """Returns the current mapping of registered face names to their class IDs."""
    return jsonify({"success": True, "class_name_to_id": class_name_to_id})


@app.route('/train_model', methods=['POST'])
def train_model():
    """Triggers the YOLOv5 model training process using a subprocess."""
    if not class_name_to_id:
        return jsonify({"success": False, "error": "No faces registered yet. Capture images first."}), 400

    if len(os.listdir(IMAGES_TRAIN_DIR)) == 0:
        return jsonify({"success": False, "error": "No images captured yet for training. Capture images first."}), 400

    try:
        update_data_yaml()

        epochs = 35 # Increased epochs slightly, 10 was too low for a good model.
        img_size = 640
        batch_size = 2 # Keep batch_size small if you have few images or limited VRAM
        weights_path = 'yolov5s.pt' # Start from pre-trained YOLOv5s weights

        project_arg_name = YOLOV5_PROJECT_ARG_NAME
        run_arg_name = YOLOV5_RUN_ARG_NAME

        # Define the full path to the specific run directory
        run_output_dir = os.path.join(YOLOV5_RUNS_ROOT_DIR, project_arg_name, run_arg_name)

        # --- FIX: Delete the old run folder before starting new training ---
        if os.path.exists(run_output_dir):
            print(f"Deleting old training run directory: {run_output_dir}")
            try:
                shutil.rmtree(run_output_dir)
                print("Old run directory deleted successfully.")
            except OSError as e:
                print(f"Error deleting old run directory {run_output_dir}: {e}")
                return jsonify({"success": False, "error": f"Failed to clean up old training data: {e}"}), 500
        # --- End Fix ---

        yolov5_repo_dir = os.path.join(BASE_DIR, 'yolov5')
        train_script_path = os.path.join(yolov5_repo_dir, 'train.py')

        if not os.path.exists(train_script_path):
            return jsonify({"success": False, "error": f"train.py not found at {train_script_path}. Please ensure the ultralytics/yolov5 repository is cloned into your project directory."}), 500

        command = [
            'python',
            train_script_path,
            '--data', DATA_YAML_PATH,
            '--epochs', str(epochs),
            '--img', str(img_size),
            '--batch-size', str(batch_size),
            '--weights', weights_path,
            '--project', project_arg_name,
            '--name', run_arg_name
        ]

        if len(class_name_to_id) == 1:
            command.append('--single-cls')
        
        print(f"Executing training command: {' '.join(command)}")

        # --- MODIFICATION START ---
        process = subprocess.run(command, capture_output=True, text=True, cwd=BASE_DIR, encoding='utf-8', errors='replace')
        # --- MODIFICATION END ---

        if process.returncode == 0:
            print("Subprocess training output:\\n", process.stdout)

            time.sleep(2) # Increased sleep slightly for more robustness. Adjust if needed.

            # Re-check the path for the trained model after training
            # This path is now correctly defined at the top as CURRENT_RUN_DIR
            if not os.path.exists(TRAINED_MODEL_FULL_PATH):
                print(f"ERROR: Trained model file not found at expected path: {TRAINED_MODEL_FULL_PATH}")
                load_detection_model(DEFAULT_YOLOV5_MODEL)
                return jsonify({"success": False, "error": f"Trained model file not found after training, path: {TRAINED_MODEL_FULL_PATH}. Please check YOLOv5 output. Using default model for detection."}), 500

            load_detection_model(TRAINED_MODEL_FULL_PATH)
            
            pathlib.PosixPath = temp 
            
            return jsonify({"success": True, "message": "Model training completed successfully.", "model_path": TRAINED_MODEL_FULL_PATH})
        else:
            print("Subprocess training error:\\n", process.stderr)
            pathlib.PosixPath = temp
            return jsonify({"success": False, "error": f"Subprocess training failed: {process.stderr}"}), 500

    except Exception as e:
        print(f"An unexpected error occurred during model training setup: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

# --- Main Execution ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)