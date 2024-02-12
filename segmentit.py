import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from collections import defaultdict

track_history = defaultdict(lambda: [])

model = YOLO("yolov8n-seg.pt")   # segmentation model
cap = cv2.VideoCapture("/Users/pascal/Desktop/pedestrians.mp4")  # Corrected path to the video file
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30  # Set FPS to 30

out = cv2.VideoWriter('instance-segmentation-object-tracking.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))

while True:
    ret, im0 = cap.read()
    if not ret:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    annotator = Annotator(im0, line_width=2)

    results = model.track(im0, persist=True)

    if results[0].boxes.id is not None and results[0].masks is not None:
        masks = results[0].masks.xy
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for mask, track_id in zip(masks, track_ids):
            annotator.seg_bbox(mask=mask,
                               mask_color=colors(track_id, True),
                               track_label=str(track_id))

    out.write(im0)
    cv2.imshow("instance-segmentation-object-tracking", im0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()
