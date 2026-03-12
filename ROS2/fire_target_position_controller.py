#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32
from geometry_msgs.msg import Point, PoseStamped
import math
import time


class FireTargetPositionController(Node):
    def __init__(self):
        super().__init__('fire_target_position_controller')

        # === Subscribers ===
        self.create_subscription(Bool, '/fire_detected', self.fire_detected_callback, 10)
        self.create_subscription(Float32, '/fire_distance', self.fire_distance_callback, 10)
        self.create_subscription(Point, '/fire_position', self.fire_position_callback, 10)

        # === Publisher ===
        self.pose_pub = self.create_publisher(PoseStamped, '/mavros2/setpoint_position/local', 10)

        # === Parameters ===
        self.target_altitude = 5.0           # desired flight height (meters AGL)
        self.fire_approach_distance = 2.0    # desired distance to stay above fire
        self.center_tolerance = 80           # pixels tolerance for centering
        self.fx = 900.0                      # camera focal length x (example OAK-D)
        self.fy = 900.0                      # camera focal length y
        self.cx = 960.0                      # principal point x (for 1920x1080)
        self.cy = 540.0                      # principal point y

        # === State ===
        self.fire_detected = False
        self.fire_distance = None
        self.fire_position = None
        self.current_pose = [0.0, 0.0, self.target_altitude]

        # === Timer ===
        self.create_timer(0.2, self.control_loop)

        self.get_logger().info('🔥 Fire Target Position Controller started.')

    # === Callbacks ===
    def fire_detected_callback(self, msg):
        self.fire_detected = msg.data

    def fire_distance_callback(self, msg):
        self.fire_distance = msg.data

    def fire_position_callback(self, msg):
        self.fire_position = (msg.x, msg.y)

    # === Control Loop ===
    def control_loop(self):
        if not self.fire_detected or self.fire_distance is None or self.fire_position is None:
            return

        if self.fire_distance < 0:
            return

        # --- Convert 2D image coordinates to 3D direction ---
        u, v = self.fire_position
        Z = self.fire_distance  # in meters

        # Convert from image pixel -> camera frame
        X = (u - self.cx) * Z / self.fx
        Y = (v - self.cy) * Z / self.fy

        # Assuming drone camera is forward-facing:
        fire_pos_cam = [X, Y, Z]

        # Convert to local NED approximation:
        # For simplicity, assume drone’s forward = +X, right = +Y, down = +Z
        # (You can refine with TF if using proper transforms)
        fire_local_x = self.current_pose[0] + fire_pos_cam[2]  # forward
        fire_local_y = self.current_pose[1] - fire_pos_cam[0]  # left/right correction
        fire_local_z = self.target_altitude                    # maintain altitude

        # Compute desired position — stay at given height above fire
        desired_pose = PoseStamped()
        desired_pose.header.stamp = self.get_clock().now().to_msg()
        desired_pose.header.frame_id = "map"
        desired_pose.pose.position.x = fire_local_x
        desired_pose.pose.position.y = fire_local_y
        desired_pose.pose.position.z = fire_local_z

        # Keep yaw fixed (can be updated later)
        desired_pose.pose.orientation.w = 1.0

        self.pose_pub.publish(desired_pose)

        self.get_logger().info(
            f"🔥 Target Pose -> x:{fire_local_x:.2f}, y:{fire_local_y:.2f}, z:{fire_local_z:.2f}, dist:{self.fire_distance:.2f}"
        )

        # === Drop Trigger Condition ===
        if self.fire_distance < self.fire_approach_distance:
            self.get_logger().info("🔥 Fire directly below — Drop fireball!")
            # Here you could publish to a servo control topic or GPIO handler


def main(args=None):
    rclpy.init(args=args)
    node = FireTargetPositionController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
