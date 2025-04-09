from ultralytics import YOLO
import cv2
import torch

# Print GPU info
print("CUDA available:", torch.cuda.is_available())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name())

# Load model with GPU if available
model = YOLO(r"best.pt").to('cuda' if torch.cuda.is_available() else 'cpu')

# Open the video file
video_path = r"d:\demo\video\sleep.mp4"
cap = cv2.VideoCapture(video_path)

# Create video writer
out = cv2.VideoWriter('output.mp4', 
                     cv2.VideoWriter_fourcc(*'mp4v'), 
                     30, 
                     (int(cap.get(3)), int(cap.get(4))))

# Set starting position (optional)
start_time = 20  # seconds
fps = cap.get(cv2.CAP_PROP_FPS)
start_frame = int(start_time * fps)
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection on the frame
    results = model(frame)
    
    # Process each detection
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            # Get the label name
            name = model.names[cls]
            
            # Draw box and label
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Màu xanh lá (0,255,0)
            cv2.putText(frame,
                       f'{name} ({conf:.2f})',
                       (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5,
                       (0, 255, 0),
                       2)

    # Write and display frame
    out.write(frame)
    cv2.imshow("YOLO Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
