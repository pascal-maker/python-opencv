import supervision as sv
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES

# Load the RF-DETR model
model = RFDETRBase()

# Define callback function to annotate each frame
def callback(frame, index):
    detections = model.predict(frame, threshold=0.5)
        
    labels = [
        f"{COCO_CLASSES[class_id]} {confidence:.2f}"
        for class_id, confidence
        in zip(detections.class_id, detections.confidence)
    ]

    annotated_frame = frame.copy()
    annotated_frame = sv.BoxAnnotator().annotate(annotated_frame, detections)
    annotated_frame = sv.LabelAnnotator().annotate(annotated_frame, detections, labels)
    return annotated_frame

# Process the video with object detection
sv.process_video(
    source_path="paris.mp4",
    target_path="annotated_paris.mp4",
    callback=callback
)
