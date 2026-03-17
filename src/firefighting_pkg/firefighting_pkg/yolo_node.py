import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32
from geometry_msgs.msg import Point
import depthai as dai
import cv2
import numpy as np
import os
import threading
from ament_index_python.packages import get_package_share_directory 

class DepthYoloNode(Node):
    def __init__(self):
        super().__init__('depth_yolo_node')
    
        # === Paths ===
        pkg_share = get_package_share_directory('firefighting_pkg')
        self.YOLO_MODEL_PATH = os.path.join(pkg_share, 'models', 'best.blob')

        self.SAVE_FRAMES = True
        self.SAVE_DIR = "/ros2_ws/detections"
        self.CONF_THRESHOLD = 0.5

        if self.SAVE_FRAMES:
            os.makedirs(self.SAVE_DIR, exist_ok=True)

        # === Publishers ===
        self.pub_fire_detected = self.create_publisher(Bool, 'firedetected', 10)
        self.pub_fire_distance = self.create_publisher(Float32, 'firedistance', 10)
        self.pub_fire_position = self.create_publisher(Point, 'fireposition', 10)

        # === Initialize DepthAI Pipeline ===
        self.get_logger().info("Initializing OAK-D VPU Pipeline...")
        self.pipeline = self.create_pipeline()
        self.device = dai.Device(self.pipeline)
        
        # Output queues
        self.qRgb = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        self.qDet = self.device.getOutputQueue(name="detections", maxSize=4, blocking=False)

        self.frame_count = 0
        self.get_logger().info("✅ OAK-D Spatial AI Node Running (Headless Mode)")

    def create_pipeline(self):
        pipeline = dai.Pipeline()

        # RGB Camera
        camRgb = pipeline.create(dai.node.ColorCamera)
        camRgb.setPreviewSize(640, 640)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setInterleaved(False)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        camRgb.setFps(15)

        # Stereo Depth
        monoLeft = pipeline.create(dai.node.MonoCamera)
        monoRight = pipeline.create(dai.node.MonoCamera)
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setCamera("left")
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setCamera("right")

        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)

        # Spatial YOLO Detection Network
        spatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
        spatialDetectionNetwork.setBlobPath(self.YOLO_MODEL_PATH)
        spatialDetectionNetwork.setConfidenceThreshold(self.CONF_THRESHOLD)
        spatialDetectionNetwork.input.setBlocking(False)
        spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
        spatialDetectionNetwork.setDepthLowerThreshold(100)
        spatialDetectionNetwork.setDepthUpperThreshold(15000) # 15 meters max
        
        # YOLOv8 specific settings (Adjust numClasses if needed)
        spatialDetectionNetwork.setNumClasses(1)
        spatialDetectionNetwork.setCoordinateSize(4)

        # Linking
        camRgb.preview.link(spatialDetectionNetwork.input)
        stereo.depth.link(spatialDetectionNetwork.inputDepth)

        # XLinkOuts
        xoutRgb = pipeline.create(dai.node.XLinkOut)
        xoutRgb.setStreamName("rgb")
        spatialDetectionNetwork.passthrough.link(xoutRgb.input)

        xoutDet = pipeline.create(dai.node.XLinkOut)
        xoutDet.setStreamName("detections")
        spatialDetectionNetwork.out.link(xoutDet.input)

        return pipeline

    def process_frames(self):
        while rclpy.ok():
            inRgb = self.qRgb.tryGet()
            inDet = self.qDet.tryGet()

            if inRgb is not None and inDet is not None:
                frame = inRgb.getCvFrame()
                detections = inDet.detections
                self.frame_count += 1

                fire_detected = False
                nearest_z = -1.0
                best_x = 0.0
                best_y = 0.0

                for detection in detections:
                    # Spatial coordinates are in millimeters. Convert to meters.
                    x_m = detection.spatialCoordinates.x / 1000.0
                    y_m = detection.spatialCoordinates.y / 1000.0
                    z_m = detection.spatialCoordinates.z / 1000.0

                    fire_detected = True
                    
                    if nearest_z == -1.0 or z_m < nearest_z:
                        nearest_z = z_m
                        best_x = x_m
                        best_y = y_m

                    # Drawing logic for saved frames
                    if self.SAVE_FRAMES and self.frame_count % 15 == 0:
                        bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                        cv2.putText(frame, f"Z: {z_m:.2f}m", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # 1. Publish Bool
                msg_detect = Bool()
                msg_detect.data = fire_detected
                self.pub_fire_detected.publish(msg_detect)

                # 2. Publish Z Distance
                msg_dist = Float32()
                msg_dist.data = float(nearest_z)
                self.pub_fire_distance.publish(msg_dist)

                # 3. Publish Spatial Point (X and Y offset in meters)
                msg_pos = Point()
                msg_pos.x = float(best_x)
                msg_pos.y = float(best_y)
                msg_pos.z = float(nearest_z)
                self.pub_fire_position.publish(msg_pos)

                if self.SAVE_FRAMES and self.frame_count % 15 == 0:
                    cv2.imwrite(os.path.join(self.SAVE_DIR, f"fire_frame_{self.frame_count}.jpg"), frame)

def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

def main(args=None):
    rclpy.init(args=args)
    node = DepthYoloNode()
    thread = threading.Thread(target=node.process_frames, daemon=True)
    thread.start()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.device.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
