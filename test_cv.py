import numpy as np
from numpy.linalg import norm
import cv2 as cv
from ultralytics import YOLO
import face_recognition
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import pickle
import uuid
from concurrent.futures import ThreadPoolExecutor
# Pose estimation model — detects 17 COCO body keypoints per person
model = YOLO('yolov8n-pose.pt')
app = FaceAnalysis(name="buffalo_sc",providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1)
# COCO keypoint indices
LEFT_SHOULDER  = 5
RIGHT_SHOULDER = 6
LEFT_WRIST     = 9
RIGHT_WRIST    = 10

CONF_THRESHOLD = 0.5  # minimum keypoint confidence to trust
EMBEDDING_PATH = "face.pkl"

#get face embedding -> check if the embedding shares some of the similarity -> if same, do nothing, otherwise, save it through pickle
def handle_face_embedding(frame):
    faces = app.get(frame)
    print(f"faces detected: {len(faces)}")  # also add this
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(check_similarity, face.embedding) for face in faces]
        for f in futures:
            f.result()  # will print any hidden crash


def check_similarity(new_embedding):
    try:
        with open(EMBEDDING_PATH, 'rb') as f:
            database = pickle.load(f)
    except (FileNotFoundError, EOFError):
        database = {}

    # empty database, save immediately
    if database == {}:
        save_embedding(new_embedding)
        return

    for person_id, person_emb in database.items():
        embedding = np.array(person_emb["embedding"])  # list → numpy array
        similarity = np.dot(embedding, new_embedding / norm(new_embedding))

        if similarity > 0.5:  # match found, person already known
            return

    # no match found after checking everyone
    save_embedding(new_embedding)


def save_embedding(embedding):
    try:
        with open(EMBEDDING_PATH, 'rb') as f:
            database = pickle.load(f)
    except (FileNotFoundError, EOFError):
        database = {}

    new_id = str(uuid.uuid4())[:8]
    normalized = embedding / norm(embedding)
    database[new_id] = {"embedding": normalized.tolist()}

    with open(EMBEDDING_PATH, 'wb') as f:
        pickle.dump(database, f)

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

    results = model.predict(frame,verbose=False)
    if not results:
        continue
    annotated = results[0].plot() 

    result = results[0]
    if result.keypoints is not None and result.boxes is not None:
        for i, box in enumerate(result.boxes):
            if is_hand_raised(result.keypoints[i]):
                handle_face_embedding(frame)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv.putText(annotated, "Hand Raised!", (x1, y1 - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv.imshow('frame', annotated)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
