from ultralytics import YOLO
import cv2
import numpy as np
from pose_analysis import calculate_box_ratio, get_box_corners
from status_analyzer import StatusAnalyzer
import torch
import time  # Add this import
from emergency_signal_detector import EmergencySignalDetector

MODEL_PATH = "../model_yolo/yolo11s-pose.pt"
MODEL_FALL_PATH = "../model_yolo/best.pt"  # Changed from yolov8_fall_detection_model.pt
MODEL_SEG_PATH = "../model_yolo/yolo11n-seg.pt"  # Add new model path
MODEL_YOLO12N_PATH = "../model_yolo/yolo12n.pt"  # Add new model path
VIDEO_PATH = "D:/demo/video/fall_demo2.mp4"

# Print GPU info
print("CUDA available:", torch.cuda.is_available())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name())

# Load models with GPU
model_pose = YOLO(MODEL_PATH).to('cuda')
model_fall = YOLO(MODEL_FALL_PATH).to('cuda')
model_seg = YOLO(MODEL_SEG_PATH).to('cuda')  # Load segmentation model
model_yolo12n = YOLO(MODEL_YOLO12N_PATH).to('cuda')  # Load additional model

# Add label mapping after model loading
label_mapping = {
    'pillow': 'pillow',  # Giữ nguyên pillow
    'bed': 'bed'         # Giữ nguyên bed
}

# Add status analyzer
status_analyzer = StatusAnalyzer()

def process_pillows(frame, results_fall):
    """Process all detections from best.pt model"""
    for r in results_fall:
        boxes = r.boxes
        for box in boxes:
            # Get class
            cls = int(box.cls[0])
            # Get name and map it
            original_name = model_fall.names[cls]
            name = label_mapping.get(original_name, original_name)
            
            # Process all detections (không chỉ giới hạn ở pillow)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # Draw detection with color based on class
            color = (255, 0, 255) if name == 'pillow' else (0, 255, 0)  # Magenta for pillow, Green for bed
            
            # Draw box and label
            cv2.rectangle(frame, 
                        (x1, y1), 
                        (x2, y2), 
                        color, 2)
            
            # Add label
            cv2.putText(frame,
                      f'{name} ({conf:.2f})',
                      (x1, y1 - 10),
                      cv2.FONT_HERSHEY_SIMPLEX,
                      0.5,
                      color,
                      2)
    return frame

def process_results(frame, results_pose):
    # Process on GPU without explicit conversion - YOLO handles this internally
    results_fall = model_fall(frame)
    results_seg = model_seg(frame)
    results_yolo12n = model_yolo12n(frame)
    annotated_frame = results_pose[0].plot(conf=False, boxes=False)
    
    # Get pillow and bed detections
    pillow_boxes = []
    bed_detections = []  # Moved up to collect all bed detections
    
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
                # Draw pillow detection with magenta color
                cv2.rectangle(annotated_frame, 
                            (x1, y1), 
                            (x2, y2), 
                            (255, 0, 255), 2)
                cv2.putText(annotated_frame,
                          f'pillow ({conf:.2f})',
                          (x1, y1 - 10),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.5,
                          (255, 0, 255),
                          2)
            elif name == 'bed':
                # Add bed detection from best.pt
                bed_detections.append({
                    'box': [x1, y1, x2, y2],
                    'conf': conf,
                    'source': 'best'  # Mark source as best.pt
                })

    # Process beds from seg model
    for r in results_seg:
        boxes = r.boxes
        for box in boxes:
            if model_seg.names[int(box.cls[0])] == 'bed':
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                bed_detections.append({
                    'box': [x1, y1, x2, y2],
                    'conf': conf,
                    'source': 'seg'
                })

    # Process beds from yolo12n model
    for r in results_yolo12n:
        boxes = r.boxes
        for box in boxes:
            if model_yolo12n.names[int(box.cls[0])] == 'bed':
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                bed_detections.append({
                    'box': [x1, y1, x2, y2],
                    'conf': conf,
                    'source': 'yolo12n'
                })

    # Sort all bed detections by confidence
    bed_detections.sort(key=lambda x: x['conf'], reverse=True)
    bed_boxes = []  # Clear and refill with highest confidence detections

    # Only draw highest confidence bed detection
    if bed_detections:
        best_bed = bed_detections[0]  # Get highest confidence detection
        bed_boxes.append(best_bed['box'])
        
        # Draw best bed detection
        x1, y1, x2, y2 = best_bed['box']
        cv2.rectangle(annotated_frame, 
                    (x1, y1), 
                    (x2, y2), 
                    (0, 255, 255), 2)  # Yellow color for beds
        
        # Add bed label with source and confidence
        cv2.putText(annotated_frame,
                  f"bed-{best_bed['source']} ({best_bed['conf']:.2f})",
                  (x1, y1 - 10),
                  cv2.FONT_HERSHEY_SIMPLEX,
                  0.5,
                  (0, 255, 255),
                  2)

    # Process person detections
    boxes = results_pose[0].boxes
    if len(boxes) > 0:
        mask = boxes.conf > 0.5
        confident_boxes = boxes[mask]
        person_boxes = []
        current_state = None
        
        for box in confident_boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf.item()
            person_boxes.append([int(x1), int(y1), int(x2), int(y2)])
            
            # Calculate ratio and state as before
            ratio, state = calculate_box_ratio([x1, y1, x2, y2])
            current_state = state
            
            # Get keypoints from pose detection
            keypoints = results_pose[0].keypoints.data[0].cpu().numpy()
            
            # Get status analysis with keypoints
            status = status_analyzer.analyze_status(person_boxes, pillow_boxes, 
                                                 bed_boxes, state, keypoints)
            
            # Define color based on status and like_fall levels
            color = {
                "sleep": (255, 191, 0),     # Deep Sky Blue
                "fall": (0, 255, 255),      # Yellow - Fall chưa đủ 10s
                "fall_prepare": (0, 69, 255), # Orange - Đang trong giai đoạn chuẩn bị
                "fall_alert": (0, 0, 255),   # Red - Fall đã xác nhận
                "normal": (0, 255, 0),       # Green
                "like_fall_1": (0, 255, 255),   # Yellow
                "like_fall_2": (0, 200, 255),   # Orange-yellow
                "like_fall_3": (0, 140, 255),   # Orange
                "like_fall_4": (0, 69, 255),    # Red-orange
                "emergency": (0, 0, 255),  # Red for emergency signal
                None: (0, 255, 0)           # Default Green
            }[status]
            
            # Draw person detection with updated status
            # Draw bounding box with color based on state
            cv2.rectangle(annotated_frame, 
                         (int(x1), int(y1)), 
                         (int(x2), int(y2)), 
                         color, 2)
            
            # Get corners
            A, B, C, D = get_box_corners([x1, y1, x2, y2])
            
            # Draw corners ABCD
            for point, label in zip([A, B, C, D], ['A', 'B', 'C', 'D']):
                cv2.circle(annotated_frame, 
                          (int(point[0]), int(point[1])), 
                          5, color, -1)
                cv2.putText(annotated_frame, 
                           label, 
                           (int(point[0])+10, int(point[1])+10),
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, color, 2)
            
            # Updated label format to not show like_fall in status brackets
            status_text = f"person K({ratio:.2f}) - X({conf:.2f})"
            if "like_fall" in state:
                level = state.split("_")[-1]
                status_text += f" - {state.replace('_'+level, '')} (L{level})"
            else:
                status_text += f" - {state}"
            
            if status and "like_fall" not in state:
                status_text += f" [{status}]"
            
            cv2.putText(annotated_frame, 
                       status_text,
                       (int(x1), int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, 2)
            
            # Update warning display based on status
            if status == "fall_alert":
                if status_analyzer.emergency_active:  # Sử dụng trực tiếp từ status_analyzer
                    cv2.putText(annotated_frame,
                               "EMERGENCY SIGNAL DETECTED!",
                               (20, 80),
                               cv2.FONT_HERSHEY_SIMPLEX,
                               1,
                               (0, 0, 255),
                               2)
                elif status_analyzer.alert_ready:
                    cv2.putText(annotated_frame,
                               "FALL ALERT - EMERGENCY!",
                               (20, 80),
                               cv2.FONT_HERSHEY_SIMPLEX,
                               1,
                               (0, 0, 255),
                               2)
            elif status == "fall_prepare":
                prep_time = status_analyzer.get_preparation_time()
                cv2.putText(annotated_frame,
                           f"Preparing Alert - {prep_time:.1f}s",
                           (20, 80),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           1,
                           (0, 69, 255),
                           2)

    return annotated_frame

# Predict with the model
results = model_pose(VIDEO_PATH, stream=True)  # predict on video stream

# Thêm cấu hình CUDA cho video capture nếu có thể
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    cap = cv2.cudacodec.createVideoReader(VIDEO_PATH)
else:
    cap = cv2.VideoCapture(VIDEO_PATH)

# Set starting position to 1:40 (100 seconds)(tua video
start_time = 120 # seconds
fps = cap.get(cv2.CAP_PROP_FPS)
start_frame = int(start_time * fps)
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

out = cv2.VideoWriter('output.mp4', 
                     cv2.VideoWriter_fourcc(*'mp4v'), 
                     30, 
                     (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    # Run detection - YOLO handles GPU internally, no need for explicit conversion
    results_pose = model_pose(frame)
    
    # Process and write frame
    processed_frame = process_results(frame, results_pose)
    out.write(processed_frame)
    
    # Display frame
    cv2.imshow("Detection", processed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
