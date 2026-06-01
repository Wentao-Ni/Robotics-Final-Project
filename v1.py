#!/usr/bin/env python3
import os
import sys
import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import TwistStamped
import cv2 as cv
from cv_bridge import CvBridge
from ultralytics import YOLO
import math
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import pickle
from numpy.linalg import norm
import uuid
import threading
EMBEDDING_PATH = "/Accounts/localniw/face.pkl"
app = FaceAnalysis(name="buffalo_sc",providers=['CUDAExecutionProvider','CPUExecutionProvider'])
app.prepare(ctx_id=-1)

try:
    model_path = os.path.join(os.path.dirname(__file__), 'yolov8n-pose.pt')
    model = YOLO(model_path)
except Exception as e:
    print("YOLO moodel is not found!")
    sys.exit(1)

def handleFaceEmbedding(frame):
    face = app.get(frame)
    
    saveEmbedding(face.embedding)
    
def getFaceEmbedding(frame):
    faces = app.get(frame)
    return faces

def checkSimilarity(new_embedding):
    try:
        with open(EMBEDDING_PATH, 'rb') as f:
            database = pickle.load(f)
    except (FileNotFoundError, EOFError):
        database = {}


    for person_id, person_data in database.items():
        embedding = np.array(person_data["embedding"])
        similarity = np.dot(embedding, new_embedding / norm(new_embedding))
        if similarity > 0.5:
            return True 

    


def saveEmbedding(embedding):
    database = {}

    new_id = str(uuid.uuid4())[:8]
    normalized = embedding / norm(embedding)
    database[new_id] = {"embedding": normalized.tolist()}

    with open(EMBEDDING_PATH, 'wb') as f:
        pickle.dump(database, f)

def get_center_pixel(box):
    box = box.cpu()
    x1 = box[0,0].item()
    y1 = box[0,1].item()
    x2 = box[0,2].item()
    y2 = box[0,3].item()

    center_x,center_y = int((x1+x2)/2), int((y1+y2)/2)
    
    return center_x,center_y


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

def is_person_block(keypoints):

    if np.array_equal(keypoints[15],[0.0,0.0,0.0]) or np.array_equal(keypoints[16],[0.0,0.0,0.0]):
        return True

    else:
        return False

class VideoSubscriberNode(Node):
    def __init__(self):
        super().__init__("video_subscriber")
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            #change it to mikey!
            '/raph/oakd/rgb/preview/image_raw',
            self.image_callback,
            10
        )
        self.person_found = False
        self.should_stop = False
        self.last_known_side = None
        self.following = False   # are we in following mode
        self.target_id = None 
        self.embedding_saved = False   # ← add this
        self.saving_face = False       # ← add this
        self.avoid_direction = None 
        self.lidar_subscription = self.create_subscription(
            LaserScan,
            "/raph/scan",
            self.lidar_callback,
            10
        )
        
        self.turtleBot_cmd_pub = self.create_publisher(TwistStamped, '/raph/cmd_vel', 10)
        self.miss_count = 0
        self.MAX_MISSES = 10 
        # self.depth_sub = self.create_subscription(
        #     CompressedImage,
        #     '/leo/oakd/rgb/preview/image_raw/compressedDepth',
        #     self.depth_callback,
        #     10
        # )
    def save_face_async(self, frame):      
        self.saving_face = True
        try:
            faces = getFaceEmbedding(frame)
            if faces and len(faces) > 0 and faces[0].embedding is not None:
                saveEmbedding(faces[0].embedding)
                self.embedding_saved = True
                self.get_logger().info("Face embedding saved!")
            else:
                self.get_logger().warn("No face detected in frame")
        except Exception as e:
            self.get_logger().error(f"Face save error: {e}")
        finally:
            self.saving_face = False

    def lidar_callback(self, msg):
        front_index = int((msg.angle_max - msg.angle_min) / (4 * msg.angle_increment))
        cone_angle = math.radians(15)
        side_angle = math.radians(45)
        cone_beams = int(cone_angle / msg.angle_increment)
        side_beams = int(side_angle / msg.angle_increment)

        def get_min(start, end):
            vals = []
            for i in range(max(0, start), min(len(msg.ranges), end)):
                r = msg.ranges[i]
                if msg.range_min < r < msg.range_max:
                    vals.append(r)
            return min(vals) if vals else float('inf')

        front_min = get_min(front_index - cone_beams, front_index + cone_beams)
        left_min  = get_min(front_index + cone_beams, front_index + cone_beams + side_beams)
        right_min = get_min(front_index - cone_beams - side_beams, front_index - cone_beams)

        if front_min < 1.0:
            self.should_stop = True
            # pick the more open side to turn toward
            if left_min >= right_min:
                self.avoid_direction = 'left'
            else:
                self.avoid_direction = 'right'
        else:
            self.should_stop = False
            self.avoid_direction = None


        


 
    def image_callback(self, msg):
        if not msg:
            print("no msg found")
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Error converting Image")
            return

        results = model.track(frame, verbose=False, persist=True)

        annotated = frame.copy()
        frame_center_x = frame.shape[1] // 2
        width = frame.shape[1]
        height = frame.shape[0]
        person_found_this_frame = False  # ← reset every frame
        person_x, person_y = 0, 0
        for r in results:                                   # ← outer loop
            annotated = r.plot()
            box = r.boxes.xyxy
            if box.shape[0] == 0:
                continue
            if r.keypoints is None:
                continue

            track_ids = r.boxes.id
            people = r.keypoints.data

            for idx, person in enumerate(people):
                conf = r.boxes.conf[idx].item()
                if conf < 0.6:
                    continue

                kpts = person.cpu().numpy()
                track_id = int(track_ids[idx].item()) if track_ids is not None else None

                left_up = is_hand_raised(kpts, "left")
                right_up = is_hand_raised(kpts, "right")

                # ── Step 1: latch onto person who raised hand ──
                if (left_up or right_up) and not self.following:
                    self.following = True
                    self.target_id = track_id
                    self.miss_count = 0
                    self.get_logger().info(f"✅ Locked onto track ID: {track_id}")

                    # Save face ONCE in background — never blocks following
                    if not self.saving_face and not self.embedding_saved:
                        frame_copy = frame.copy()
                        thread = threading.Thread(
                            target=self.save_face_async,
                            args=(frame_copy,),
                            daemon=True
                        )
                        thread.start()

                # ── Step 2: follow by track ID only — no face recognition here ──
                if self.following and track_id == self.target_id:
                    person_box = r.boxes.xyxy[idx:idx+1]
                    person_x, person_y = get_center_pixel(person_box)
                    person_found_this_frame = True
                    cv.circle(annotated, (person_x, person_y), 10, (0, 0, 255), 5)
                    break

                # ── Step 3: re-identify ONLY after track ID fully lost ──
                elif self.following and self.miss_count > self.MAX_MISSES:
                    if not self.saving_face:  # don't re-id while saving
                        person_box = r.boxes.xyxy[idx:idx+1]
                        x1 = max(0, int(person_box[0, 0].item()))
                        y1 = max(0, int(person_box[0, 1].item()))
                        x2 = min(frame.shape[1], int(person_box[0, 2].item()))
                        y2 = min(frame.shape[0], int(person_box[0, 3].item()))
                        face_crop = frame[y1:y2, x1:x2]

                        if face_crop.size > 0:
                            faces = getFaceEmbedding(face_crop)
                            if faces and len(faces) > 0 and faces[0].embedding is not None:
                                if checkSimilarity(faces[0].embedding):
                                    self.target_id = track_id
                                    self.miss_count = 0
                                    self.get_logger().info(f"Re-identified! New ID: {track_id}")

        # ← everything below is OUTSIDE both loops
        if not person_found_this_frame:
            self.miss_count += 1
            if self.miss_count > self.MAX_MISSES:
                self.following = False
                self.target_id = None
                self.miss_count = 0

        else:
            self.miss_count = 0

        # ← build twist command
        twistS = TwistStamped()
        twistS.header.stamp = self.get_clock().now().to_msg()
        twistS.header.frame_id = 'base_link'
        angular_gain = 0.005
        max_linear_speed = 0.4

        if person_found_this_frame:
            error_x = float(person_x) - float(width) / 2.0
            twistS.twist.angular.z = float(-error_x * angular_gain)
            twistS.twist.linear.x = float(max_linear_speed)

        # elif self.last_known_side == 'left':
        #     twistS.twist.angular.z = 0.3
        #     twistS.twist.linear.x = 0.0
        #     cv.putText(annotated, "Searching left...", (50, 100),
        #             cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # elif self.last_known_side == 'right':
        #     twistS.twist.angular.z = -0.3
        #     twistS.twist.linear.x = 0.0
        #     cv.putText(annotated, "Searching right...", (50, 100),
        #             cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        # elif self.following and self.miss_count <= self.MAX_MISSES:
        #     # Still following, briefly lost — keep last command or slow down
        #     twistS.twist.linear.x = float(max_linear_speed * 0.5)
        #     twistS.twist.angular.z = 0.0  # go straight while searching

        window_name = "Hand Raise Detection"
        cv.imshow(window_name, annotated)
        cv.waitKey(1)

        if not self.should_stop:
            # Normal following
            if person_found_this_frame:
                error_x = float(person_x) - float(width) / 2.0
                twistS.twist.angular.z = float(-error_x * angular_gain)
                twistS.twist.linear.x = float(max_linear_speed)
            else:
                twistS.twist.linear.x = 0.0
                twistS.twist.angular.z = 0.0

            self.turtleBot_cmd_pub.publish(twistS)

        # else:
        #     # Obstacle detected — actively avoid it
        #     avoid = TwistStamped()
        #     avoid.header.stamp = self.get_clock().now().to_msg()
        #     avoid.header.frame_id = 'base_link'

        #     if self.avoid_direction == 'left':
        #         avoid.twist.angular.z =  0.5
        #         avoid.twist.linear.x  =  0.1   # creep forward while turning
        #         self.get_logger().info("Obstacle! Avoiding left...")
        #     elif self.avoid_direction == 'right':
        #         avoid.twist.angular.z = -0.5
        #         avoid.twist.linear.x  =  0.1
        #         self.get_logger().info("Obstacle! Avoiding right...")
        #     else:
        #         avoid.twist.linear.x  = 0.0    # fully boxed in, stop
        #         avoid.twist.angular.z = 0.0

        #     self.turtleBot_cmd_pub.publish(avoid)

def main(args=None):
    rclpy.init(args=args)
    node = VideoSubscriberNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()
        cv.destroyAllWindows()
        rclpy.shutdown()
    


if __name__ == '__main__':
    main()
