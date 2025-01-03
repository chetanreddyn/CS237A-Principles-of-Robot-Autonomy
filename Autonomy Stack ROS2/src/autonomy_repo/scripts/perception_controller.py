#!/usr/bin/env python3

import rclpy
from asl_tb3_lib.control import BaseController
from asl_tb3_msgs.msg import TurtleBotControl
from std_msgs.msg import Bool


class PerceptionController(BaseController):
    
    def __init__(self):
        super().__init__("perception_controller")
        self.declare_parameter("active",True)
        
        self.detection_sub = self.create_subscription(Bool,"/detector_bool",self.detection_callback,10)
        self.time_active_last = self.get_clock().now().nanoseconds/1e9
        self.last_detection_state = None
        
    def detection_callback(self, msg: Bool):
        current_time  = self.get_clock().now().nanoseconds/1e9
        if msg.data and self.last_detection_state == False:  # if flip false to true (detected)
            self.set_parameters([rclpy.Parameter("active",value=False)])
        elif current_time - self.time_active_last>=5:            
            self.set_parameters([rclpy.Parameter("active",value=True)])
        self.last_detection_state = msg.data

    @property
    def active(self) -> bool:
        return self.get_parameter("active").value
    
    def compute_control(self) -> TurtleBotControl:
        
        control = TurtleBotControl()
        
        if self.active:
            self.time_active_last = self.get_clock().now().nanoseconds/1e9
            control.omega = 0.5
        else:
            control.omega=0.0
        
        return control
    
if __name__ == "__main__":
    rclpy.init()
    perception_controller = PerceptionController()
    rclpy.spin(perception_controller)
    rclpy.shutdown()
    