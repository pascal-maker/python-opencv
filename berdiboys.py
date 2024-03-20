import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolov8-soccer.pt")

# Video paths
MODEL_PATH = '/Users/pascal/desktop/best.pt'
video_path = '/Users/pascal/Desktop/soccer.mp4'
output_video_path = '/Users/pascal/Desktop/soccer_output.mp4'

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

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
