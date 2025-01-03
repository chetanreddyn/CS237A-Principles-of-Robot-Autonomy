#!/usr/bin/env python3

import sys, select, termios, tty

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class KeyboardTeleopNode(Node):
    def __init__(self, settings):
        super().__init__('keyboard_teleop_node')
        
        self.msg = """
        Control Your Turtlebot!
        ---------------------------
        Moving around:
        u    i    o
        j    k    l
        m    ,    .
        
        q/z : increase/decrease max speeds by 10%
        w/x : increase/decrease only linear speed by 10%
        e/c : increase/decrease only angular speed by 10%
        space key, k : force stop
        anything else : stop smoothly

        CTRL-C to quit
        """
        
        self.moveBindings = {
            'i': (1, 0),
            'o': (1, -1),
            'j': (0, 1),
            'l': (0, -1),
            'u': (1, 1),
            ',': (-1, 0),
            '.': (-1, 1),
            'm': (-1, -1),
        }
        
        self.speedBindings = {
            'q': (1.1, 1.1),
            'z': (0.9, 0.9),
            'w': (1.1, 1),
            'x': (0.9, 1),
            'e': (1, 1.1),
            'c': (1, 0.9),
        }
        
        self.settings = settings
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 5)
        self.run()

    def getKey(self):
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''

        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def vels(self, speed, turn):
        return f"Currently: speed => {speed:.2f}, turn => {turn:.2f}"
    
    def run(self):
        '''
        Initial params
        '''
        speed = 0.1
        turn = 0.3
        x = 0.0
        th = 0.0
        status = 0
        count = 0
        target_speed = 0.0
        target_turn = 0.0
        control_speed = 0.0
        control_turn = 0.0
        
        try:
            print(self.msg)
            print(self.vels(speed, turn))
            
            while(1):
                key = self.getKey()
                if key in self.moveBindings.keys():
                    x = self.moveBindings[key][0]
                    th = self.moveBindings[key][1]
                    count = 0
                    
                elif key in self.speedBindings.keys():
                    speed = speed * self.speedBindings[key][0]
                    turn = turn * self.speedBindings[key][1]
                    count = 0
                    print(self.vels(speed, turn))
                    
                    if status == 14:
                        print(self.msg)
                    status = (status + 1) % 15
                    
                elif key == ' ' or key == 'k':
                    x = 0.0
                    th = 0.0
                    control_speed = 0.0
                    control_turn = 0.0
                    
                else:
                    count = count + 1
                    if count > 4:
                        x = 0.0
                        th = 0.0
                    if (key == '\x03'):
                        break

                target_speed = speed * x
                target_turn = turn * th

                if target_speed > control_speed:
                    control_speed = min(target_speed, control_speed + 0.02)
                elif target_speed < control_speed:
                    control_speed = max(target_speed, control_speed - 0.02)
                else:
                    control_speed = target_speed

                if target_turn > control_turn:
                    control_turn = min(target_turn, control_turn + 0.1)
                elif target_turn < control_turn:
                    control_turn = max(target_turn, control_turn - 0.1)
                else:
                    control_turn = target_turn

                twist = Twist()
                twist.linear.x = control_speed; twist.linear.y = 0.0; twist.linear.z = 0.0
                twist.angular.x = 0.0; twist.angular.y = 0.0; twist.angular.z = control_turn
                self.cmd_vel_publisher.publish(twist)
                
        except Exception as e:
            print(e)

        finally:
            twist = Twist()
            twist.linear.x = 0.0; twist.linear.y = 0.0; twist.linear.z = 0.0
            twist.angular.x = 0.0; twist.angular.y = 0.0; twist.angular.z = 0.0
            self.cmd_vel_publisher.publish(twist)


def main(args=None):
    settings = termios.tcgetattr(sys.stdin)
    rclpy.init(args=args)
    keyboard_teleop_node = KeyboardTeleopNode(settings=settings)
    rclpy.spin(keyboard_teleop_node)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
