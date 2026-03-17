#!/usr/bin/env python3
import time
import threading
import rclpy
import math
from rclpy.node import Node
from std_msgs.msg import Bool, Float32
from geometry_msgs.msg import Point
from firefighting_interfaces.msg import MissionState
from firefighting_interfaces.srv import StartMission, RTL
from pymavlink import mavutil

class MavlinkControllerNode(Node):
    def __init__(self):
        super().__init__('mavlink_controller_node')
        
        # --- Parameters ---
        self.declare_parameter('connection', '/dev/ttyAMA0')
        self.declare_parameter('baudrate', 57600)
        self.declare_parameter('search_alt', 5.0)          
        self.declare_parameter('drop_alt', 5.0)            
        self.declare_parameter('spiral_spacing', 8.0)      
        self.declare_parameter('spiral_max_radius', 100.0) 
        
        self.connection_str = self.get_parameter('connection').get_parameter_value().string_value
        self.baudrate = self.get_parameter('baudrate').get_parameter_value().integer_value
        self.search_alt = self.get_parameter('search_alt').get_parameter_value().double_value
        self.drop_alt = self.get_parameter('drop_alt').get_parameter_value().double_value
        self.spiral_spacing = self.get_parameter('spiral_spacing').get_parameter_value().double_value
        self.max_radius = self.get_parameter('spiral_max_radius').get_parameter_value().double_value
        
        # --- State Variables ---
        self.phase = "IDLE"
        self.target_lat = 0.0
        self.target_lon = 0.0
        self.waypoints = [] 
        self.wp_index = 0
        
        self.fire_gps_lat = 0.0
        self.fire_gps_lon = 0.0
        
        self.fire_detected = False
        self.fire_dist = -1.0 # Z distance
        self.fire_off_x = 0.0 # X distance
        self.fire_off_y = 0.0 # Y distance
        self.detection_count = 0

        # --- Continuous Telemetry Variables ---
        self.current_lat = 0.0
        self.current_lon = 0.0
        self.current_alt = 0.0
        self.current_hdg = 0.0
        self.system_status = 0
        
        # --- ROS Interfaces ---
        self.pub_state = self.create_publisher(MissionState, '/mission_state', 10)
        self.sub_detect = self.create_subscription(Bool, 'firedetected', self.cb_detect, 10)
        self.sub_dist = self.create_subscription(Float32, 'firedistance', self.cb_dist, 10)
        self.sub_pos = self.create_subscription(Point, 'fireposition', self.cb_pos, 10)
        
        self.srv_start = self.create_service(StartMission, 'start_mission', self.cb_start)
        self.srv_rtl = self.create_service(RTL, 'rtl', self.cb_rtl)
        
        # --- MAVLink Connection & Threads ---
        self.master = None
        self.connect_mavlink()
        
        # Dedicated thread to constantly read noisy serial data
        self.telem_thread = threading.Thread(target=self.telemetry_loop, daemon=True)
        self.telem_thread.start()

        # Logic thread
        self.logic_thread = threading.Thread(target=self.mavlink_loop, daemon=True)
        self.logic_thread.start()
        
        self.get_logger().info('✅ Physical Drone Controller Ready')

    # --- Callbacks ---
    def cb_start(self, req, res):
        if self.phase != "IDLE":
            res.success = False; res.status = "Busy"; return res
        
        self.target_lat = req.latitude
        self.target_lon = req.longitude
        self.waypoints = self.generate_spiral_points(self.target_lat, self.target_lon)
        self.wp_index = 0
        self.detection_count = 0
        
        self.phase = "ARMING"
        res.success = True; res.status = "Mission Started"
        return res

    def cb_rtl(self, req, res):
        self.phase = "RTB"
        res.success = True
        return res

    def cb_detect(self, msg): self.fire_detected = msg.data
    def cb_dist(self, msg): self.fire_dist = msg.data
    def cb_pos(self, msg): 
        self.fire_off_x = msg.x
        self.fire_off_y = msg.y

    # --- Geometry ---
    def generate_spiral_points(self, center_lat, center_lon):
        wps = []
        b = self.spiral_spacing / (2 * math.pi)
        theta = 0.0
        max_theta = self.max_radius / b
        while theta < max_theta:
            r = b * theta
            dn = r * math.cos(theta)
            de = r * math.sin(theta)
            d_lat = dn / 111320.0
            d_lon = de / (111320.0 * math.cos(math.radians(center_lat)))
            wps.append((center_lat + d_lat, center_lon + d_lon))
            if r < 1.0: theta += 0.5
            else: theta += (5.0 / r) 
        return wps

    def calculate_fire_gps(self):
        # Using continuous telemetry data instead of blocking read
        if self.current_lat == 0.0: return False
        
        # Simplified geospatial projection from camera spatial offsets
        azimuth = self.current_hdg + math.atan2(self.fire_off_x, self.fire_dist)
        ground_dist = math.sqrt(self.fire_dist**2 + self.fire_off_x**2)
        
        delta_n = ground_dist * math.cos(azimuth)
        delta_e = ground_dist * math.sin(azimuth)
        
        self.fire_gps_lat = self.current_lat + (delta_n / 111320.0)
        self.fire_gps_lon = self.current_lon + (delta_e / (111320.0 * math.cos(math.radians(self.current_lat))))
        
        self.get_logger().info(f"🎯 FIRE LOCKED AT GPS: {self.fire_gps_lat:.6f}, {self.fire_gps_lon:.6f}")
        return True

    # --- Dedicated Telemetry Loop (CRITICAL FOR HARDWARE) ---
    def telemetry_loop(self):
        while rclpy.ok():
            msg = self.master.recv_match(blocking=True, timeout=0.1)
            if not msg: continue
            
            msg_type = msg.get_type()
            if msg_type == 'GLOBAL_POSITION_INT':
                self.current_lat = msg.lat / 1e7
                self.current_lon = msg.lon / 1e7
                self.current_alt = msg.relative_alt / 1000.0
                self.current_hdg = math.radians(msg.hdg / 100.0)
            elif msg_type == 'HEARTBEAT':
                self.system_status = msg.system_status

    # --- Main State Machine ---
    def mavlink_loop(self):
        # Lock Gripper on boot
        self.trigger_servo(1900) 
        
        while rclpy.ok():
            # Robust Fire Accumulator
            if self.phase == "SCANNING":
                if self.fire_detected and self.fire_dist > 0:
                    self.detection_count += 2 
                else:
                    self.detection_count = max(0, self.detection_count - 1) 
                
                if self.detection_count >= 5: # Real-world requires higher confidence
                    self.get_logger().warn("🔥 FIRE VERIFIED! BRAKING.")
                    self.send_velocity(0,0,0) # Stop momentum
                    time.sleep(1.5) # Allow drone to stabilize
                    self.phase = "APPROACHING"

            # Phase Execution
            if self.phase == "IDLE":
                pass
                
            elif self.phase == "ARMING":
                self.get_logger().info("Waiting for EKF & GPS 3D Fix...")
                # Wait until ArduPilot says it is ready to arm
                while self.system_status != mavutil.mavlink.MAV_STATE_STANDBY:
                    time.sleep(1)
                
                self.set_mode("GUIDED")
                self.arm_drone()
                self.phase = "TAKEOFF"
                
            elif self.phase == "TAKEOFF":
                self.takeoff(self.search_alt)
                while not self.check_altitude(self.search_alt, 1.0):
                    time.sleep(0.5)
                self.phase = "NAVIGATE_TO_ZONE"
                
            elif self.phase == "NAVIGATE_TO_ZONE":
                self.goto_global(self.target_lat, self.target_lon, self.search_alt)
                if self.check_reached_global(self.target_lat, self.target_lon, 3.0):
                    self.phase = "SCANNING"
                    
            elif self.phase == "SCANNING":
                if self.wp_index < len(self.waypoints):
                    lat, lon = self.waypoints[self.wp_index]
                    self.goto_global(lat, lon, self.search_alt)
                    if self.check_reached_global(lat, lon, 2.5):
                        self.wp_index += 1
                else:
                    self.phase = "RTB"
                    
            elif self.phase == "APPROACHING":
                if self.calculate_fire_gps():
                    self.phase = "GLIDING"
                else:
                    self.condition_yaw(15) # Spin to search
                    time.sleep(0.5)
                    
            elif self.phase == "GLIDING":
                self.goto_global(self.fire_gps_lat, self.fire_gps_lon, self.search_alt)
                if self.check_reached_global(self.fire_gps_lat, self.fire_gps_lon, 1.0):
                    self.phase = "DESCENDING"
                    
            elif self.phase == "DESCENDING":
                self.goto_global(self.fire_gps_lat, self.fire_gps_lon, self.drop_alt)
                if self.check_altitude(self.drop_alt, 0.5):
                    self.phase = "DROP"
                    
            elif self.phase == "DROP":
                self.get_logger().info("💣 DROPPING PAYLOAD")
                self.trigger_servo(1100) # Open Gripper
                time.sleep(3)
                self.phase = "RTB"
                
            elif self.phase == "RTB":
                self.set_mode("RTL")
                self.phase = "IDLE"
                    
            time.sleep(0.1)

    # --- MAVLink Commands ---
    def connect_mavlink(self):
        self.get_logger().info(f'Connecting to {self.connection_str}...')
        self.master = mavutil.mavlink_connection(self.connection_str, baud=self.baudrate)
        self.master.wait_heartbeat()
        self.get_logger().info('MAVLink Connected.')

    def arm_drone(self):
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
        self.master.motors_armed_wait()

    def set_mode(self, mode):
        if mode in self.master.mode_mapping():
            self.master.mav.set_mode_send(
                self.master.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                self.master.mode_mapping()[mode])

    def trigger_servo(self, pwm):
        # 9 = AUX OUT 1 on Pixhawk
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_SERVO, 0, 9, pwm, 0, 0, 0, 0, 0)

    def takeoff(self, alt):
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, alt)

    def goto_global(self, lat, lon, alt):
        self.master.mav.set_position_target_global_int_send(
            0, self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            0b0000111111111000, int(lat*1e7), int(lon*1e7), alt, 0, 0, 0, 0, 0, 0, 0, 0)

    def send_velocity(self, vx, vy, vz):
        self.master.mav.set_position_target_local_ned_send(
            0, self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            0b0000111111000111, 0, 0, 0, vx, vy, vz, 0, 0, 0, 0, 0)

    def condition_yaw(self, rate):
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_CONDITION_YAW, 0, 0, rate, 1, 1, 0, 0, 0)

    def check_reached_global(self, lat, lon, threshold):
        if self.current_lat == 0.0: return False
        d = math.sqrt((self.current_lat - lat)**2 + (self.current_lon - lon)**2) * 111320
        return d < threshold

    def check_altitude(self, target_alt, threshold):
        if self.current_alt == 0.0: return False
        return abs(self.current_alt - target_alt) < threshold

def main(args=None):
    rclpy.init(args=args)
    node = MavlinkControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
