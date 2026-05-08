import numpy as np
import cv2 as cv
from ultralytics import YOLO
import face_recognition
# Pose estimation model — detects 17 COCO body keypoints per person
model = YOLO('yolov8n-pose.pt')

# COCO keypoint indices
LEFT_SHOULDER  = 5
RIGHT_SHOULDER = 6
LEFT_WRIST     = 9
RIGHT_WRIST    = 10

CONF_THRESHOLD = 0.5  # minimum keypoint confidence to trust


def is_hand_raised(kps):
    """Return True if either wrist keypoint is above its shoulder (lower y value)."""
    kp = kps.data[0]  # shape (17, 3): x, y, confidence
    for wrist_idx, shoulder_idx in [(LEFT_WRIST, LEFT_SHOULDER), (RIGHT_WRIST, RIGHT_SHOULDER)]:
        if kp[wrist_idx, 2] > CONF_THRESHOLD and kp[shoulder_idx, 2] > CONF_THRESHOLD:
            if kp[wrist_idx, 1] < kp[shoulder_idx, 1]:  # y increases downward
                return True
    return False


cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    results = model.track(frame,show=True, persist=True)
    annotated = results[0].plot() 

    result = results[0]
    if result.keypoints is not None and result.boxes is not None:
        for i, box in enumerate(result.boxes):
            if is_hand_raised(result.keypoints[i]):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv.putText(annotated, "Hand Raised!", (x1, y1 - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv.imshow('frame', annotated)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
