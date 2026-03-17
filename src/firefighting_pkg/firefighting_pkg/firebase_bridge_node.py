import rclpy
from rclpy.node import Node
import firebase_admin
import json
from std_msgs.msg import String
from firebase_admin import credentials, firestore
from firefighting_interfaces.srv import StartMission, RTL
from firefighting_interfaces.msg import MissionState
import os 

class FirebaseBridge(Node):
    def __init__(self):
        super().__init__('firebase_bridge_node')
        
        # Path to key (Confirm this matches your Pi's path)
        key_path = os.path.expanduser('/home/drone/ros2_ws/src/firefighting_pkg/config/firebase_key.json')
        
        try:
            cred = credentials.Certificate(key_path)
            firebase_admin.initialize_app(cred)
            self.db = firestore.client()
            self.get_logger().info('Firebase initialized.')
        except Exception as e:
            self.get_logger().error(f'Firebase Error: {e}')
            return

        self.client = self.create_client(StartMission, 'start_mission')
        self.rtl_client = self.create_client(RTL, 'rtl')

        # Subscribe to Telemetry
        self.sub_telemetry = self.create_subscription(String, '/drone_telemetry', self.telemetry_cb, 10)
        self.sub_state = self.create_subscription(MissionState, '/mission_state', self.mission_state_cb, 10)
        
        self.mission_ref = self.db.collection('missions')
        self.status_ref = self.db.collection('drone_status').document('drone_01')
        self.query_watch = self.mission_ref.on_snapshot(self.on_snapshot)
        
        self.current_mission_id = None
        self.is_busy = False

    def on_snapshot(self, col_snapshot, changes, read_time):
        for change in changes:
            data = change.document.to_dict()
            status = data.get('status')
            doc_id = change.document.id
            
            # Deploy
            if status == 'IN_PROGRESS' and not self.is_busy:
                self.get_logger().info(f"New Mission: {doc_id}")
                self.current_mission_id = doc_id
                self.trigger_drone(data)
                
            # Abort
            elif status in ['abort_signal', 'ABORTED', 'CANCELLED']:
                if self.current_mission_id == doc_id or self.is_busy:
                    self.get_logger().warn(f"Abort triggered for {doc_id}")
                    self.trigger_rtl()

    def telemetry_cb(self, msg):
        try:
            data = json.loads(msg.data)
            data["last_update"] = firestore.SERVER_TIMESTAMP
            self.status_ref.set(data, merge=True)
        except Exception as e:
            pass

    def mission_state_cb(self, msg):
        if msg.state_description != "IDLE":
            self.is_busy = True
        elif msg.state_description == "IDLE" and self.is_busy:
            self.get_logger().info("Mission Completed/Reset.")
            self.is_busy = False
            
            if self.current_mission_id:
                try:
                    self.mission_ref.document(self.current_mission_id).update({
                        'status': 'COMPLETED',
                        'progress': 100
                    })
                    self.current_mission_id = None
                except Exception as e:
                    pass

    def trigger_drone(self, data):
        req = StartMission.Request()
        req.latitude = float(data['lat'])
        req.longitude = float(data['lng'])
        req.altitude_msl = 5.0 
        self.client.call_async(req)

    def trigger_rtl(self):
        req = RTL.Request()
        self.rtl_client.call_async(req)

def main(args=None):
    rclpy.init(args=args)
    node = FirebaseBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
