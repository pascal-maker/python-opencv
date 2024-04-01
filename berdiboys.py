import cv2
from ultralytics import YOLO
from roboflow import Roboflow
from IPython.display import Image, display

# Define model path, video path, and output video path
MODEL_PATH = '/Users/pascal/desktop/yolov8m-football-1.pt'
VIDEO_PATH = '/Users/pascal/Desktop/soccer.mp4'
OUTPUT_VIDEO_PATH = '/Users/pascal/Desktop/soccer_output.mp4'


# Train the YOLO model with the specified dataset
model = YOLO("yolov8m-football.pt")
results = model.train( epochs=10, imgsz=320)

# Load the trained YOLO model
model = YOLO(MODEL_PATH)

# Open the video file
cap = cv2.VideoCapture(VIDEO_PATH)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, frame_rate, (frame_width, frame_height))

# Loop through the video frames
while cap.isOpened():
    success, frame = cap.read()
    if success:
        # Perform inference and plot detections
        results = model(frame, verbose=False, conf=0.5)
        annotated_frame = results[0].plot()

        # Write annotated frame to the output video
        out.write(annotated_frame)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Handle user input for quitting
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
