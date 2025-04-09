from flask import Flask, Response, request, jsonify
from flask_sock import Sock
from ultralytics import YOLO
import cv2
import numpy as np
from module.pose_analysis import calculate_box_ratio, get_box_corners
from module.status_analyzer import StatusAnalyzer
import time
from module.emergency_signal_detector import EmergencySignalDetector
import io
import base64
import json
import os

app = Flask(__name__)
sock = Sock(app)

# Update model paths
MODEL_PATH = "E:/fuji/demo/model_yolo/yolo11s-pose.pt"
MODEL_FALL_PATH = "E:/fuji/demo/model_yolo/best.pt"
MODEL_SEG_PATH = "E:/fuji/demo/model_yolo/yolo11n.pt"
MODEL_YOLO12N_PATH = "E:/fuji/demo/model_yolo/yolo11n.pt"

def verify_model_paths():
    """Verify all model paths exist"""
    paths = [MODEL_PATH, MODEL_FALL_PATH, MODEL_SEG_PATH, MODEL_YOLO12N_PATH]
    for path in paths:
        if not os.path.exists(path):
            print(f"Không tìm thấy file model tại: {path}")
            return False
        print(f"Đã tìm thấy file model tại: {path}")
    return True

# Check model paths before loading
if not verify_model_paths():
    print("Vui lòng kiểm tra lại đường dẫn các file model")
    exit(1)

# Load models without GPU
model_pose = YOLO(MODEL_PATH)
model_fall = YOLO(MODEL_FALL_PATH)
model_seg = YOLO(MODEL_SEG_PATH)
model_yolo12n = YOLO(MODEL_YOLO12N_PATH)

# Initialize status analyzer
status_analyzer = StatusAnalyzer()

# Label mapping remains the same
label_mapping = {
    'pillow': 'pillow',
    'bed': 'bed'
}

def process_pillows(frame, results_fall):
    # ...existing code...
    return frame

def process_results(frame, results_pose):
    # Process without GPU specifications
    results_fall = model_fall(frame)
    results_seg = model_seg(frame)
    results_yolo12n = model_yolo12n(frame)
    annotated_frame = results_pose[0].plot(conf=False, boxes=False)
    
    pillow_boxes = []
    bed_detections = []
    
    # Prepare detection results dictionary
    detection_results = {
        'beds': [],
        'pillows': [],
        'detections': []
    }
    
    # Process detections from best.pt model
    for r in results_fall:
        boxes = r.boxes
        for box in boxes:
            original_name = model_fall.names[int(box.cls[0])]
            name = label_mapping.get(original_name, original_name)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            if name == 'pillow':
                pillow_boxes.append([x1, y1, x2, y2])
                detection_results['pillows'].append([x1, y1, x2, y2])
                # ... Drawing code for pillows ...
            elif name == 'bed':
                bed_detections.append({
                    'box': [x1, y1, x2, y2],
                    'conf': conf,
                    'source': 'best'
                })

    # ... Rest of the processing code ...

    # Update detection results
    if bed_detections:
        best_bed = bed_detections[0]
        detection_results['beds'].append(best_bed['box'])

    # Process person detections and update detection_results['detections']
    boxes = results_pose[0].boxes
    if len(boxes) > 0:
        mask = boxes.conf > 0.5
        confident_boxes = boxes[mask]
        for box in confident_boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = float(box.conf[0])
            ratio, state = calculate_box_ratio([x1, y1, x2, y2])
            
            keypoints = results_pose[0].keypoints.data[0].numpy()  # Changed from cpu()
            status = status_analyzer.analyze_status(
                [[int(x1), int(y1), int(x2), int(y2)]], 
                pillow_boxes, 
                detection_results['beds'], 
                state, 
                keypoints
            )
            
            detection_results['detections'].append({
                'box': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': float(conf),
                'ratio': float(ratio),
                'state': state,
                'status': status,
                'is_emergency': status_analyzer.emergency_active
            })

    return annotated_frame, detection_results

def process_frame(frame):
    """Process a single frame"""
    results_pose = model_pose(frame)
    return process_results(frame, results_pose)

@app.route('/process_video', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    video_file = request.files['video']
    temp_path = "temp_video.mp4"
    video_file.save(temp_path)
    
    cap = cv2.VideoCapture(temp_path)
    processed_frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        processed_frame, _ = process_frame(frame)
        _, buffer = cv2.imencode('.jpg', processed_frame)
        processed_frames.append(base64.b64encode(buffer).decode('utf-8'))
    
    cap.release()
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    return jsonify({
        'frames': processed_frames,
        'frame_count': len(processed_frames)
    })

@app.route('/process_stream', methods=['POST'])
def process_stream():
    if 'frame' not in request.files:
        return jsonify({'error': 'No frame provided'}), 400
    
    frame_file = request.files['frame']
    frame_bytes = frame_file.read()
    frame_np = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
    
    processed_frame, _ = process_frame(frame)
    
    _, buffer = cv2.imencode('.jpg', processed_frame)
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        'processed_frame': frame_base64
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'gpu_available': False,  # Changed to False since we're using CPU
        'models_loaded': True
    })

def find_virtual_camera():
    """Find virtual camera with more robust error handling"""
    try:
        # Try different camera indices with DirectShow
        for i in range(10):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    cap.release()
                    return i, cv2.CAP_DSHOW
            cap.release()
        
        # If DirectShow fails, try default backend
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    cap.release()
                    return i, None
            cap.release()
                
        return None, None
    except Exception as e:
        print(f"Error finding camera: {e}")
        return None, None

@sock.route('/ws')
def handle_websocket(ws):
    cap = None
    camera_idx = None
    backend = None
    reconnect_delay = 1
    max_reconnect_delay = 30
    
    try:
        camera_idx, backend = find_virtual_camera()
        if camera_idx is None:
            ws.send(json.dumps({'error': 'No virtual camera found'}))
            return

        while True:
            try:
                if cap is None or not cap.isOpened():
                    if backend:
                        cap = cv2.VideoCapture(camera_idx, backend)
                    else:
                        cap = cv2.VideoCapture(camera_idx)
                        
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
                    cap.set(cv2.CAP_PROP_FPS, 30)

                ret, frame = cap.read()
                if not ret or frame is None:
                    raise Exception("Failed to grab frame")

                reconnect_delay = 1  # Reset delay on success

                # Process frame and get results
                processed_frame, detection_results = process_frame(frame)
                
                # Encode frame with reduced quality for better performance
                _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')

                # Prepare response
                result = {
                    'image': frame_base64,
                    'dimensions': {'width': 960, 'height': 540},
                    'emergency_alerts': [],
                    **detection_results
                }

                # Add emergency alerts if needed
                if status_analyzer.emergency_active:
                    result['emergency_alerts'].append({
                        'type': 'emergency',
                        'message': 'EMERGENCY SIGNAL DETECTED!'
                    })
                for detection in detection_results.get('detections', []):
                    if detection.get('status') == 'fall_alert':
                        result['emergency_alerts'].append({
                            'type': 'fall',
                            'message': 'FALL ALERT - EMERGENCY!'
                        })
                        break

                ws.send(json.dumps(result))
                time.sleep(1/30)  # Limit to 30 FPS for CPU performance

            except Exception as e:
                print(f"Stream error: {e}")
                if cap:
                    cap.release()
                    cap = None
                time.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        if cap:
            cap.release()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
