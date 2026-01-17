import pathlib
import platform
import torch
import cv2
import numpy as np

# Path fix for Windows
plt = platform.system()
if plt == 'Windows': pathlib.PosixPath = pathlib.WindowsPath

# 1. Load Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best (1).pt')

# 2. Set Confidence Threshold (Inside or right before the loop)
model.conf = 0.60  # Only show detections with > 60% certainty
model.iou = 0.45   # Intersection over Union (handles overlapping boxes)

cap = cv2.VideoCapture(0)
print("")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # 3. Inference
    results = model(frame)

    # 4. Filter or Process Results
    # If you want to see what class IDs are being detected in the console:
    # print(results.pandas().xyxy[0]) 

    # 5. Render and Show
    annotated_frame = np.squeeze(results.render())
    cv2.imshow('YOLOv5 Custom Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('ctrl+c'):
        break

cap.release()
cv2.destroyAllWindows()