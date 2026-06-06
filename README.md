# Human-Following Robot

A ROS2 node for a TurtleBot 4 that follows a person who raises their hand. It uses YOLOv8-pose to detect people and hand raises, InsightFace to save/re-identify the target's face, and LiDAR for obstacle detection. All logic lives in one node, `VideoSubscriberNode`, split into an `image_callback` (detect, lock onto target, follow) and a `lidar_callback` (obstacle detection).

## Dependencies

- ROS2 Jazzy (`rclpy`, `sensor_msgs`, `geometry_msgs`, `cv_bridge`)
- `numpy`
- `opencv-python` (cv2)
- `ultralytics` (YOLOv8)
- `insightface`
- `onnxruntime`

Install Python dependencies:

```bash
pip install numpy opencv-python ultralytics insightface onnxruntime
```

Also required:
- `yolov8n-pose.pt` in the same directory as the node.
- A writable path for `EMBEDDING_PATH` (default `/Accounts/localniw/face.pkl`) to save the facial embedding of the target person.

## Build

From the workspace root:

```bash
cd ~/ros2_ws
colcon build
source install/setup.bash
```

## Run

```bash
ros2 run visual_ackermann_package camera_subscriber_node
```

A window titled "Hand Raise Detection" will pop up showing the live camera feed, annotated with detected people (with their bounding boxes, a red circle indicating the center of the bounding box). 
Stop with `Ctrl+C`.

## Topics

- Subscribes: `/raph/oakd/rgb/preview/image_raw`, `/raph/scan`
- Publishes: `/raph/cmd_vel`

## AI Statement
Claude Code was used for bug-fixing and for generating library-specific syntax throughout this project. Especially for formatting the face embedding dictionary before pickling, the conditions of the face detection guard in the save_face_async function. Claude was also used to suggest ways to improve the formatting of this README.md. 

## Note
Obstacle avoidance movement is commented out (left in to show effort); the robot detects obstacles but stops rather than driving around them.
