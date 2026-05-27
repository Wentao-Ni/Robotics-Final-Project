#!/usr/bin/env python3
import os
import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2 as cv
from cv_bridge import CvBridge
from ultralytics import YOLO

try:
    model_path = os.path.join(os.path.dirname(__file__), 'yolov8n-pose.pt')
    model = YOLO(model_path)
except Exception as e:
    print("YOLO moodel is not found!")

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

class VideoSubscriberNode(Node):
    def __init__(self):
        super().__init__("video_subscriber")
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            #change it to mikey!
            '/leo/oakd/rgb/preview/image_raw',
            self.image_callback,
            10
        )
    

    def image_callback(self,msg):
        if not msg:
            print("no msg found")
        try:
            #convered to the cv2 image type
            frame = self.bridge.imgmsg_to_cv2(msg,desired_encoding="bgr8")
            
        except Exception as e:
            self.get_logger().error(f"Error converting Image")
        
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
                    cv.putText(
                        annotated, label, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                    )

        cv.imshow("Hand Raise Detection", frame)


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
