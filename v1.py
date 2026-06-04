#=============================
#HUMAN FOLLOWING NODE
#this node contains all subscribers and publishers used by our human following robot
#Claude code was used for bug-fixing, as well as generating certain lines containing
#library-specific syntax. Citations are present before those lines.
#=============================

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

#=============================
#FACIAL RECOGNITION FUNCTIONS/INITIALIZATION
#facial recognition did not have a large usecase in our demo, but with obstacle
#avoidance working the following functions would be necessary
#=============================

#where the facial embedding should be stored
EMBEDDING_PATH = "/Accounts/localniw/face.pkl"

#initializing the face analysis app
app = FaceAnalysis(name="buffalo_sc",providers=['CUDAExecutionProvider','CPUExecutionProvider'])
app.prepare(ctx_id=-1)

#add a face embedding to face.pkl
def handleFaceEmbedding(frame):
    face = app.get(frame)
    saveEmbedding(face.embedding)

#get all faces in frame    
def getFaceEmbedding(frame):
    faces = app.get(frame)
    return faces

#check the similarity between a new face found and the one in the database (face.pkl)
def checkSimilarity(new_embedding):
    try:
        with open(EMBEDDING_PATH, 'rb') as f:
            database = pickle.load(f)
    except (FileNotFoundError, EOFError):
        database = {}
    for person_data in database.items():
        #CITATION: Claude code was used to generate the following two lines
        embedding = np.array(person_data["embedding"]) #reformat the embedding already in face.pkl into an np.array
        similarity = np.dot(embedding, new_embedding / norm(new_embedding))  #this uses cos(theta) = a.d/|a|.|b| to calculate the similarity

        #if cos(theta) = 0.5 or more, then the two vector representations of the embeddings are at least in the same quadrant,
        # meaning they are similar enough to be considered the same.
        if similarity > 0.5:
            return True 

#save an embedding in face.pkl
def saveEmbedding(embedding):
    #wipe the database each time, because we only want the face of the person who
    #first raised their hand during the run
    database = {}

    new_id = str(uuid.uuid4())[:8] #give this embedding a unique ID
    #normalize the embedding so the cosine similarity check works as intended later
    normalized = embedding / norm(embedding)
    #CITATION: the following line was generated using Claude code, we found format-handling quite unintuitive
    database[new_id] = {"embedding": normalized.tolist()}

    with open(EMBEDDING_PATH, 'wb') as f:
        pickle.dump(database, f) #pickle.dump converts the facial embedding data structure into a byte stream to add to a file


#=============================
#POSE-HANDLING/COMPUTER VISION FUNCTIONS/INITILIALIZATION
#=============================

#initialize pose-tracking/human-recognition app
try:
    #CITATION: the following line was generated using Claude
    model_path = os.path.join(os.path.dirname(__file__), 'yolov8n-pose.pt') #yolov8n-pose.pt contains the pre-trained model
    model = YOLO(model_path)
except Exception as e:
    print("YOLO moodel is not found!")
    sys.exit(1)

#function to find the center pixel of a bounding box
def get_center_pixel(box):
    box = box.cpu() 
    x1 = box[0,0].item()
    y1 = box[0,1].item()
    x2 = box[0,2].item()
    y2 = box[0,3].item()

    #calculate the center
    center_x,center_y = int((x1+x2)/2), int((y1+y2)/2)
    return center_x,center_y

#function to check whether the hand is raised
def is_hand_raised(keypoints, side="left"):

    #in YOLO pose-tracking, there are 17 keypoints, each representing a different part of the human body
    if side == "left":
        shoulder = keypoints[5]
        wrist = keypoints[9]   #left wrist
    else:
        shoulder = keypoints[6]
        wrist = keypoints[10]  #right wrist

    #each keypoint also carries a confidence value, we should only return that the
    #the hand is raised if we are reasonably confident on keypoint locations
    if shoulder[2] < 0.5 or wrist[2] < 0.5:
        return False

    #return whether the wrist is above the shoulder on the chosen side
    return wrist[1] < shoulder[1]

#check if a person is blocked by an obstacle
def is_person_block(keypoints):
    #this function is simple; it simply checks whether the ankle keypoints are invisible for the target,
    #meaning an obstacle is partially obstructing view of them and we need to go around it
    if np.array_equal(keypoints[15],[0.0,0.0,0.0]) or np.array_equal(keypoints[16],[0.0,0.0,0.0]):
        return True
    else:
        return False


#=============================
#VIDEO AND LIDAR SUBSCRIBER NODE
#we recognize that putting both all subscribers in one class is sloppy, please bear with us here
#=============================

#CITATION: the general structure for subsriber/publisher initilization and logic was
#taken from in-class labs, courtesy of Dr. Chelsey Edge and Carleton College. The same is true
#for callback logic.
class VideoSubscriberNode(Node):
    def __init__(self):
        super().__init__("video_subscriber")
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/raph/oakd/rgb/preview/image_raw', #raph was the robot used in our final demo
            self.image_callback,
            10
        ) #initialize camera subscription

        self.lidar_subscription = self.create_subscription(
            LaserScan,
            "/raph/scan",
            self.lidar_callback,
            10
        ) #initialize lidar subscription

        self.turtleBot_cmd_pub = self.create_publisher(
            TwistStamped, 
            '/raph/cmd_vel', 
            10) #initialize cmd_vel publisher

        #initializing various variables that control what "state" the robot is in and other function logic
        self.following = False
        self.person_found = False
        self.should_stop = False
        self.last_known_side = None
        self.target_id = None 
        self.embedding_saved = False
        self.saving_face = False
        self.avoid_direction = None 
        self.miss_count = 0
        self.MAX_MISSES = 10 
        
    #this function saves a face present in the frame, the frame given to this function as an argument will
    #be cropped to only the bounding box of the person with their hand currently raised
    def save_face_async(self, frame):      
        self.saving_face = True #enter the saving_face state
        try:
            faces = getFaceEmbedding(frame)
            if faces and len(faces) > 0 and faces[0].embedding is not None: #CITATION: the conditions of this if statement were debugged by Claude
                saveEmbedding(faces[0].embedding)
                self.embedding_saved = True
                self.get_logger().info("Face embedding saved!")
            else:
                self.get_logger().warn("No face detected in frame")
        except Exception as e:
            self.get_logger().error(f"Face save error: {e}")
        finally: #finally is used here to ensure that the saving_face state is exited regardless of exception
            self.saving_face = False 

    #=============================
    #LIDAR CALLBACK
    #=============================

    def lidar_callback(self, msg):

        #trig to find the portion of lidar ranges corresponding to the front and side sections
        front_index = int((msg.angle_max - msg.angle_min) / (4 * msg.angle_increment))
        cone_angle = math.radians(15)
        side_angle = math.radians(45)
        cone_beams = int(cone_angle / msg.angle_increment)
        side_beams = int(side_angle / msg.angle_increment)

        #this functino finds the smallest range among a given selection of lidar ranges
        def get_min(start, end):
            vals = []
            for i in range(max(0, start), min(len(msg.ranges), end)):
                r = msg.ranges[i]
                if msg.range_min < r < msg.range_max: #adds only valid ranges to the final minimum check
                    vals.append(r)
            #return infinity if there were no valid ranges in this section
            return min(vals) if vals else float('inf')

        #get minimum ranges for each section
        front_min = get_min(front_index - cone_beams, front_index + cone_beams)
        left_min  = get_min(front_index + cone_beams, front_index + cone_beams + side_beams)
        right_min = get_min(front_index - cone_beams - side_beams, front_index - cone_beams)

        #if the fron is blocked, start avoiding the blockage depending on which side has more open space
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
 
    #=============================
    #IMAGE CALLBACK
    #=============================

    def image_callback(self, msg):
        if not msg:
            print("no msg found")
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Error converting Image")
            return

        #this function, built into the yolo pose model, does the brunt of the computational work.
        #It creates persistent bounding boxes around the humans in each frame it receives from the 
        #camera and save each frame's data
        results = model.track(frame, verbose=False, persist=True)

        #this is the frame that shows on the monitor with the bounding box and labels drawn
        annotated = frame.copy()

        frame_center_x = frame.shape[1] // 2
        width = frame.shape[1]
        height = frame.shape[0]
        person_found_this_frame = False
        person_x, person_y = 0, 0

        #for each frame returned by the model.track function, create an annotated version and also
        #follow the target or change states as necessary
        for r in results:
            annotated = r.plot()
            box = r.boxes.xyxy

            #skip this iteration if no humans are present
            if box.shape[0] == 0:
                continue
            if r.keypoints is None:
                continue

            track_ids = r.boxes.id
            people = r.keypoints.data

            #for each person present in the frame
            for idx, person in enumerate(people):

                #ignore if confidence value is too low
                conf = r.boxes.conf[idx].item()
                if conf < 0.6:
                    continue

                kpts = person.cpu().numpy()
                #CITATION: the following line was rewritten with Claude code after bugs found
                track_id = int(track_ids[idx].item()) if track_ids is not None else None

                #check if this person has a hand up
                left_up = is_hand_raised(kpts, "left")
                right_up = is_hand_raised(kpts, "right")

                #if they do and the following state has not been entered, enter it and save this person's face embedding
                if (left_up or right_up) and not self.following:
                    self.following = True
                    self.target_id = track_id
                    self.miss_count = 0 #at the start of the following phase, the target has been missing from the fram 0 times

                    #if the face is not already actively being saved and has not already been saved, save it
                    if not self.saving_face and not self.embedding_saved:
                        frame_copy = frame.copy()
                        #CITATION: the following line was generated using Claude code
                        thread = threading.Thread(
                            target=self.save_face_async,
                            args=(frame_copy,),
                            daemon=True
                        )
                        thread.start()

                #if the target is actively being followed
                if self.following and track_id == self.target_id:
                    person_box = r.boxes.xyxy[idx:idx+1]
                    person_x, person_y = get_center_pixel(person_box)
                    person_found_this_frame = True
                    #CITATION: this and other lines relating to specific formatting for
                    #annotating the frame were generated by Claude code
                    cv.circle(annotated, (person_x, person_y), 10, (0, 0, 255), 5)
                    break

                #if the target has been lost for more frames than specificed in MAX_MISSES
                elif self.following and self.miss_count > self.MAX_MISSES:
                    if not self.saving_face:  #if the person's face is actively being saved, ignore this for now
                        person_box = r.boxes.xyxy[idx:idx+1]
                        #the mathematical logic for cropping the frame to the person was generated by Claude code
                        x1 = max(0, int(person_box[0, 0].item()))
                        y1 = max(0, int(person_box[0, 1].item()))
                        x2 = min(frame.shape[1], int(person_box[0, 2].item()))
                        y2 = min(frame.shape[0], int(person_box[0, 3].item()))
                        #create the cropped version of the frame as specified in the save embedding function
                        face_crop = frame[y1:y2, x1:x2]

                        #if the face cropped to is similar to the target's, reset the miss count and resume following
                        if face_crop.size > 0:
                            faces = getFaceEmbedding(face_crop)
                            if faces and len(faces) > 0 and faces[0].embedding is not None:
                                if checkSimilarity(faces[0].embedding):
                                    self.target_id = track_id
                                    self.miss_count = 0

        #regardless of state, if the target is not found, increase miss_count and check if the person relocation state needs to be entered
        if not person_found_this_frame:
            self.miss_count += 1
            if self.miss_count > self.MAX_MISSES:
                self.following = False
                self.target_id = None
                self.miss_count = 0
        else:
            self.miss_count = 0

        #CITATION: the logic for calculating angular gain was taken from the cv light-following lab,
        #courtesy of Chelsey Edge and Carleton College
        twistS = TwistStamped()
        twistS.header.stamp = self.get_clock().now().to_msg()
        twistS.header.frame_id = 'base_link'
        angular_gain = 0.005
        max_linear_speed = 0.4

        #calculating the anglular gain to match the target's movement when the robot is in the
        #following state without issue.
        if not self.should_stop:
            if person_found_this_frame:
                error_x = float(person_x) - float(width) / 2.0
                twistS.twist.angular.z = float(-error_x * angular_gain)
                twistS.twist.linear.x = float(max_linear_speed)
            else:
                twistS.twist.linear.x = 0.0
                twistS.twist.angular.z = 0.0

            self.turtleBot_cmd_pub.publish(twistS)

        #display the annotated frame on the monitor
        window_name = "Hand Raise Detection"
        cv.imshow(window_name, annotated)
        cv.waitKey(1)


        #=============================
        #OBSTACLE AVOIDANCE LOGIC 
        #this is not currently working, but the code is left commented to show evidence of effort.
        #=============================

        # elif self.last_known_side == 'left':
        #     twistS.twist.angular.z = 0.3
        #     twistS.twist.linear.x = 0.0
        #     cv.putText(annotated, "Searching left...", (50, 100),
        #             cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        #
        # elif self.last_known_side == 'right':
        #     twistS.twist.angular.z = -0.3
        #     twistS.twist.linear.x = 0.0
        #     cv.putText(annotated, "Searching right...", (50, 100),
        #             cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        #
        # elif self.following and self.miss_count <= self.MAX_MISSES:
        #     twistS.twist.linear.x = float(max_linear_speed * 0.5)
        #     twistS.twist.angular.z = 0.0  # go straight while searching
        #
        # else:
        #     avoid = TwistStamped()
        #     avoid.header.stamp = self.get_clock().now().to_msg()
        #     avoid.header.frame_id = 'base_link'
        #
        #     if self.avoid_direction == 'left':
        #         avoid.twist.angular.z =  0.5
        #         avoid.twist.linear.x  =  0.1
        #         self.get_logger().info("Avoiding left")
        #
        #     elif self.avoid_direction == 'right':
        #         avoid.twist.angular.z = -0.5
        #         avoid.twist.linear.x  =  0.1
        #         self.get_logger().info("Avoiding right")
        #
        #     else:
        #         avoid.twist.linear.x  = 0.0
        #         avoid.twist.angular.z = 0.0
        #
        #     self.turtleBot_cmd_pub.publish(avoid)

#the main function simply starts the node and shuts everything down upon failure
def main(args=None):
    rclpy.init(args=args)
    node = VideoSubscriberNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt: #look for keyboard singal to stop
        node.destroy_node()
        cv.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
