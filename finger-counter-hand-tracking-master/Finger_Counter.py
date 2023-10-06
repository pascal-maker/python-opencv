import cv2
import time
import os
import HandTracking as ht

#############################################################
cam_width, cam_height = 640, 480
previous_time = 0
#############################################################

# Create the 'finger_image' directory if it doesn't exist
folder_path = "finger_image"
if not os.path.exists(folder_path):
    os.mkdir(folder_path)

my_list = os.listdir(folder_path)
overlay_list = []

for path in my_list:
    image = cv2.imread(f'{folder_path}/{path}')
    overlay_list.append(image)

detector = ht.HandDetector(detection_confidence=0.70)

tip_id = [4, 8, 12, 16, 20]

cap = cv2.VideoCapture(0)
cap.set(3, cam_width)
cap.set(4, cam_height)

while cap.isOpened():
    success, img = cap.read()
    img = detector.find_hands(img)

    landmark_list, _ = detector.find_position(img, draw=False)

    if len(landmark_list) != 0:
        fingers = detector.fingers_up()

        total_fingers = fingers.count(1)

        if total_fingers < len(overlay_list):
            h, w, c = overlay_list[total_fingers].shape
            img[0:h, 0:w] = overlay_list[total_fingers]
        else:
            # Handle the case when total_fingers is out of range
            # You can display a message or take appropriate action
            pass  # Replace 'pass' with your desired error handling or action

        cv2.rectangle(img, (20, 300), (170, 475), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, text=str(total_fingers), org=(50, 450), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=8,
                    color=(255, 0, 0), thickness=20)

    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    cv2.putText(img, text=f"FPS : {int(fps)}", org=(400, 70), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2,
                color=(255, 0, 0), thickness=3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
