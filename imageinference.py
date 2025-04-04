import supervision as sv
from PIL import Image
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES

# Load model
model = RFDETRBase()

# Load local image (no need for requests or io)
image = Image.open("pascal.jpg")

# Run object detection
detections = model.predict(image, threshold=0.5)

# Generate labels for each detection
labels = [
    f"{COCO_CLASSES[class_id]} {confidence:.2f}"
    for class_id, confidence
    in zip(detections.class_id, detections.confidence)
]

# Annotate image with boxes and labels
annotated_image = image.copy()
annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

# Display the annotated image
sv.plot_image(annotated_image)
