import cv2
from sort import *
import math
import numpy as np
from ultralytics import YOLO

# Define the body part indices
body_index = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

# Body part colors
part_colors = {
    "nose": (255, 0, 0),
    "left_eye": (0, 255, 0),
    "right_eye": (0, 0, 255),
    "left_ear": (255, 255, 0),
    "right_ear": (255, 0, 255),
    "left_shoulder": (0, 255, 255),
    "right_shoulder": (255, 255, 255),
    "left_elbow": (128, 128, 0),
    "right_elbow": (0, 128, 128),
    "left_wrist": (128, 0, 128),
    "right_wrist": (0, 128, 0),
    "left_hip": (128, 0, 0),
    "right_hip": (0, 128, 0),
    "left_knee": (0, 0, 128),
    "right_knee": (128, 0, 0),
    "left_ankle": (0, 0, 0),
    "right_ankle": (255, 255, 255),
}

cap = cv2.VideoCapture('/Users/pascal/Desktop/ghentbycar.mp4')
model_object_detection = YOLO('yolov8n.pt')
model_pose_detection = YOLO("yolov8s-pose.pt")

classnames = ['car', 'bus', 'truck', 'person']
tracker = Sort(max_age=20)
line = [320, 350, 620, 350]

while 1:
    ret, frame = cap.read()
    if not ret:
        cap = cv2.VideoCapture('/Users/pascal/Desktop/ghentbycar.mp4')
        continue
    detections = np.empty((0, 5))
    
    # Use object detection model for all objects by default
    model = model_object_detection

    result = model(frame, stream=1)
    for info in result:
        boxes = info.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            classindex = box.cls[0]
            conf = math.ceil(conf * 100)
            classindex = int(classindex)
            
            # Check if classindex is within the valid range
            if 0 <= classindex < len(classnames):
                objectdetect = classnames[classindex]

                # Lower the confidence threshold to 30%
                if (objectdetect == 'car' or objectdetect == 'bus' or objectdetect == 'truck' or
                        objectdetect == 'person') and conf > 30:
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    new_detections = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, new_detections))

                    # Use pose detection model only when the detected object is a person
                    if objectdetect == 'person':
                        print("Using pose detection model")
                        model = model_pose_detection
                        # Run pose detection on the region containing the person
                        pose_results = model(frame[y1:y2, x1:x2])
                        
                        # Visualize keypoints with different colors
                        for pose_result in pose_results:
                            keypoints = pose_result.keypoints.data[0]
                            for i, (x, y, _) in enumerate(keypoints):
                                x, y = int(x) + x1, int(y) + y1
                                color = part_colors.get(list(body_index.keys())[i], (0, 255, 0))  # Default to green if color not defined
                                cv2.circle(frame, (x, y), 5, color, -1)

    track_result = tracker.update(detections)
    cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 255, 255), 7)

    for results in track_result:
        x1, y1, x2, y2, id = results
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)

        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2

        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        if detections.size > 0 and id < len(detections):
            cv2.putText(frame, f'{id} Conf: {detections[id][4]}%', (x1 + 8, y1 - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if line[0] < cx < line[2] and line[1] - 20 < cy < line[1] + 20:
            cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 15)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)
