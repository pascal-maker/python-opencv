import torch
import torchvision
import cv2

# Load a pre-trained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Open the video
video_path = '/Users/pascal/Desktop/boxingvideo.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to a PyTorch tensor
    frame_tensor = torchvision.transforms.ToTensor()(frame).unsqueeze(0)

    # Pass the frame through the model to detect objects
    with torch.no_grad():
        prediction = model(frame_tensor)

    # Extract the bounding boxes and labels from the prediction
    boxes = prediction[0]['boxes'].numpy().astype(int)
    labels = prediction[0]['labels'].numpy()

    # Draw bounding boxes on the frame
    for box in boxes:
        x, y, w, h = box
        frame = cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)

    cv2.imshow('Video Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
