#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from pymavlink import mavutil
import time

class PreFlightCheckNode(Node):
    def __init__(self):
        super().__init__('pre_flight_check_node')
        
        self.get_logger().info("=== RUNNING PRE-FLIGHT HARDWARE DIAGNOSTICS ===")
        
        # 1. Check Pixhawk MAVLink
        self.check_pixhawk('/dev/ttyAMA0', 57600)
        
        # 2. Listen for OAK-D
        self.fire_sub = self.create_subscription(Bool, 'firedetected', self.camera_cb, 10)
        self.camera_online = False
        self.get_logger().info("⏳ Waiting for OAK-D camera topic...")

    def check_pixhawk(self, port, baud):
        self.get_logger().info(f"⏳ Testing Pixhawk connection on {port}...")
        try:
            master = mavutil.mavlink_connection(port, baud=baud)
            master.wait_heartbeat(timeout=5)
            self.get_logger().info("✅ PIXHAWK: Heartbeat received. UART connection is solid.")
            
            # Request GPS status
            msg = master.recv_match(type='GPS_RAW_INT', blocking=True, timeout=3)
            if msg:
                if msg.fix_type >= 3:
                    self.get_logger().info(f"✅ GPS: 3D Fix acquired ({msg.satellites_visible} satellites).")
                else:
                    self.get_logger().warn(f"⚠️ GPS: No 3D fix yet (Type: {msg.fix_type}). Cannot arm safely.")
            master.close()
        except Exception as e:
            self.get_logger().error(f"❌ PIXHAWK: Connection failed! Check wiring and Docker --device flags. Error: {e}")

    def camera_cb(self, msg):
        if not self.camera_online:
            self.camera_online = True
            self.get_logger().info("✅ OAK-D CAMERA: Topic 'firedetected' is actively publishing.")
            self.get_logger().info("=== DIAGNOSTICS COMPLETE. PRESS CTRL+C TO EXIT ===")

def main(args=None):
    rclpy.init(args=args)
    node = PreFlightCheckNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()