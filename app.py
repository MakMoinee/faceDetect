from flask import Flask, render_template_string, request, jsonify
import torch
from PIL import Image
import io
import base64
import numpy as np
import cv2

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
app = Flask(__name__)

# Load YOLOv5 model - change path to your weights
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt', force_reload=False)

@app.route('/')
def index():
    return "<h1>Go to /stream to start camera detection</h1>"

@app.route('/stream')
def stream():
    # Serve a page with video + button + canvas for drawing boxes
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
                    // Optional: check camera permission status first
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
                            })
                       
                    }

                    // Request camera access (this triggers the permission prompt if needed)
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

                // Draw video frame to an offscreen canvas
                const offscreen = document.createElement('canvas');
                offscreen.width = video.videoWidth;
                offscreen.height = video.videoHeight;
                const offctx = offscreen.getContext('2d');
                offctx.drawImage(video, 0, 0, offscreen.width, offscreen.height);

                // Get image as JPEG base64
                const dataUrl = offscreen.toDataURL('image/jpeg', 0.7);
                const base64 = dataUrl.split(',')[1];

                // Send frame to server for detection
                try {
                    const response = await fetch('/detect_frame', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ image: base64 })
                    });
                    const data = await response.json();

                    // Clear canvas and draw bounding boxes
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

                // Process next frame after a short delay to reduce load
                setTimeout(detectFrame, 200);
            }
        </script>

    </body>
    </html>
    """)

@app.route('/detect_frame', methods=['POST'])
def detect_frame():
    try:
        data = request.get_json()
        img_b64 = data['image']
        img_bytes = base64.b64decode(img_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        results = model(img)
        detections = results.pandas().xyxy[0]

        faces = []
        for _, row in detections.iterrows():
            if (row['name']!="person"):
                return
            
            faces.append({
                "xmin": float(row['xmin']),
                "ymin": float(row['ymin']),
                "xmax": float(row['xmax']),
                "ymax": float(row['ymax']),
                "confidence": float(row['confidence']),
                "class": int(row['class']),
                "name": row['name']
            })

        return jsonify({"faces": faces})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Run on all interfaces so accessible on LAN
    app.run(host='0.0.0.0', port=5000)
