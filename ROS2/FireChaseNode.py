#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32
from geometry_msgs.msg import Point, Twist

class FireChaseController(Node):
    def __init__(self):
        super().__init__('fire_chase_controller')

        # --- Subscribers ---
        self.create_subscription(Bool, '/fire_detected', self.fire_detected_callback, 10)
        self.create_subscription(Float32, '/fire_distance', self.fire_distance_callback, 10)
        self.create_subscription(Point, '/fire_position', self.fire_position_callback, 10)

        # --- Publisher to MAVROS2 velocity setpoints ---
        self.vel_pub = self.create_publisher(Twist, '/mavros2/setpoint_velocity/cmd_vel_unstamped', 10)

        # --- Optional publisher for dropping mechanism ---
        # self.drop_pub = self.create_publisher(Bool, '/drop_fireball', 10)

        # --- Parameters ---
        self.target_distance = 2.0        # meters – desired distance from fire
        self.center_tolerance = 60        # pixels
        self.speed_gain = 0.5             # base linear speed
        self.angular_gain = 0.002         # pixel→yaw conversion gain

        # --- State variables ---
        self.fire_detected = False
        self.fire_distance = None
        self.fire_position = None
        self.frame_width = 1920
        self.frame_height = 1080

        self.get_logger().info('🔥 Fire Chase Controller started.')

        # Control loop timer
        self.create_timer(0.1, self.control_loop)

    # === Callback functions ===
    def fire_detected_callback(self, msg):
        self.fire_detected = msg.data

    def fire_distance_callback(self, msg):
        self.fire_distance = msg.data

    def fire_position_callback(self, msg):
        self.fire_position = (msg.x, msg.y)

    # === Control loop ===
    def control_loop(self):
        if not self.fire_detected or self.fire_distance is None or self.fire_position is None:
            return  # no valid data yet

        x_center, y_center = self.fire_position
        img_center_x = self.frame_width / 2
        img_center_y = self.frame_height / 2

        # Compute offset from image center
        x_error = img_center_x - x_center
        y_error = img_center_y - y_center

        # Forward/backward velocity control based on distance
        distance_error = self.fire_distance - self.target_distance

        twist = Twist()

        # Move forward if fire is far
        if abs(distance_error) > 0.3:
            twist.linear.x = -self.speed_gain * distance_error  # negative because drone faces forward
        else:
            twist.linear.x = 0.0

        # Align left/right
        if abs(x_error) > self.center_tolerance:
            twist.angular.z = self.angular_gain * x_error
        else:
            twist.angular.z = 0.0

        # Up/down correction (optional)
        if abs(y_error) > self.center_tolerance:
            twist.linear.z = -0.0015 * y_error
        else:
            twist.linear.z = 0.0

        # Publish velocity command
        self.vel_pub.publish(twist)

        self.get_logger().info(
            f"Fire: {self.fire_distance:.2f} m | "
            f"x_err: {x_error:.0f} | y_err: {y_error:.0f} | "
            f"cmd vx: {twist.linear.x:.2f}, vz: {twist.linear.z:.2f}, yaw: {twist.angular.z:.2f}"
        )

        # If close enough and centered → drop
        if self.fire_distance < 1.5 and abs(x_error) < self.center_tolerance and abs(y_error) < self.center_tolerance:
            self.get_logger().info("🔥 Target reached — drop payload!")
            # drop_msg = Bool()
            # drop_msg.data = True
            # self.drop_pub.publish(drop_msg)

def main(args=None):
    rclpy.init(args=args)
    node = FireChaseController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
