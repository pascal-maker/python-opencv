import cv2
from ultralytics import YOLO
import numpy as np

# Load the YOLO pose model
model = YOLO("yolov8-soccer.pt")

# Video path
video_path = '/Users/pascal/Desktop/fcberdi.mp4'

# Open the video file
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()
    if success:
        # Perform inference and plot detections
        results = model(frame, verbose=False, conf=0.5)
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Handle user input for quitting
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
