import cv2
from ultralytics import YOLO

#this is the pretrained pose-tracker model from YOLO
model = YOLO("yolov8n-pose.pt")

#initialize video capture
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

#Function to recognize when hand is raised: 
# receives pose tracking keypoints (e.g. elbow, wrist) and what side is being analyzed
def is_hand_raised(keypoints, side="left"):

    #in YOLO, there are 17 keypoints, each representing a different part of the human body
    if side == "left":
        shoulder = keypoints[5]
        wrist = keypoints[9]   # left wrist
    else:
        shoulder = keypoints[6]
        wrist = keypoints[10]  # right wrist

    #each keypoint also carries a confidence value, we should only return that the
    #the hand is raised if we are reasonably confident on keypoint locations
    if shoulder[2] < 0.5 or wrist[2] < 0.5:
        return False

    #return whether the wrist is above the shoulder on the chosen side
    return wrist[1] < shoulder[1]


while True:
    ret, frame = cap.read()

    #if the frame capture could not be initialized
    if not ret:
        print("Failed to grab frame")
        break

    #this is the YOLO function doing the heavy lifting. It extends normal object detection by assigning a unique ID 
    # to each detected object, allowing you to track movement of people over time. It returns a list of frames, each containing
    #bounding boxes, track IDs, confidence scores, and class IDs (what the object is)
    results = model.track(frame, verbose=False, persist=True)

    annotated = frame.copy()

    for r in results:
        #plot of the results on the frame
        annotated = r.plot()

        #no poseable objects in frame
        if r.keypoints is None:
            continue

        people = r.keypoints.data

        for person in people:
            # Convert to numpy-like structure: [x, y, conf]
            kpts = person.cpu().numpy()

            left_up = is_hand_raised(kpts, "left")
            right_up = is_hand_raised(kpts, "right")

            label = ""

            if left_up and right_up:
                label = "Both hands raised"
            elif left_up:
                label = "Left hand raised"
            elif right_up:
                label = "Right hand raised"

            if label:
                #place the label on the frame
                cv2.putText(
                    annotated, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                )

    cv2.imshow("Hand Raise Detection", annotated)

    #quit on "q" key pressed
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
