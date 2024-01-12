from roboflow import Roboflow
import clearml
from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
import numpy as np
from IPython.display import Image, display
import io

# Download the dataset stored on Roboflow
rf = Roboflow(api_key="lvJtfzYNSRQrGmcNtdUd")
project = rf.workspace().project("fcberdi")
model = project.version(1).model
dataset = project.version(1).download("yolov8")

# Connecting ClearML with the current process (Colab notebook)
clearml.browser_login()

# Define model & data path
model = YOLO('yolov8n.pt')
DATASET_PATH = "/Users/pascal/Desktop/fcberdi.v1i.yolov8/data.yaml"

# Train the model with CUDA or MPS
results = model.train(data=DATASET_PATH, epochs=10, imgsz=320)

# Define video path & model path
MODEL_PATH = '/Users/pascal/desktop/best.pt'
VIDEO_PATH = '/Users/pascal/desktop/fcberdi.mp4'
OUTPUT_VIDEO_PATH = '/Users/pascal/desktop/fcberdi_output.mp4'

def detect_on_frame(model_path, video_path, frame_number, conf_threshold=0.35):
    """Extracts a specific frame from a video, runs YOLOv8 detection on it, and displays the result.

    Args:
        model_path (str): Path to the YOLO model file.
        video_path (str): Path to the input video file.
        frame_number (int): The frame number to extract and process.
        conf_threshold (float, optional): Confidence threshold for YOLO model detection. Defaults to 0.35.
    """
    
    model = YOLO(model_path)
    
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, frame = cap.read()
    cap.release()

    if success:
        # Run YOLOv8 inference on the frame and visualize the results
        results = model(frame, conf=conf_threshold)
        annotated_frame = results[0].plot()
        _, encoded_image = cv2.imencode('.png', annotated_frame)
        ipy_img = Image(data=encoded_image.tobytes())
        display(ipy_img)
    else:
        print(f"Failed to read the frame at position {frame_number} from the video.")

# Detect on a single frame from the video
FRAME_NUMBER = 500
detect_on_frame(MODEL_PATH, VIDEO_PATH, FRAME_NUMBER, 0.5)

def process_and_detect_on_video(model_path, video_path, output_path=None, display_video=False, conf_threshold=0.5):
    """Process a video file with YOLO model, optionally save and display the output.

    Args:
        model_path (str): Path to the YOLO model file.
        video_path (str): Path to the input video file.
        output_path (str, optional): Path where the output video will be saved. If None, the video won't be saved.
        display_video (bool): Whether to display the video during processing.
        conf_threshold (float): Confidence threshold for YOLO model detection.
    """

    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    out = None
    if output_path:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, frame_rate, (frame_width, frame_height))

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            results = model(frame, conf=conf_threshold)
            annotated_frame = results[0].plot()

            if out:
                out.write(annotated_frame)

            if display_video:
                cv2.imshow("YOLOv8 Inference", annotated_frame)
                # Press Q on the keyboard to exit when displaying the video
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        cap.release()
        if out:
            out.release()
        if display_video:
            cv2.destroyAllWindows()

# Process and detect on the video
process_and_detect_on_video(MODEL_PATH, VIDEO_PATH, OUTPUT_VIDEO_PATH, display_video=False, conf_threshold=0.5)