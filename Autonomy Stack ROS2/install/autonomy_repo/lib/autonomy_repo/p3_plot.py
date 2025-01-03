#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import rclpy
from rclpy.node import Node
from rclpy.task import Future

from asl_tb3_msgs.msg import TurtleBotState
from asl_tb3_lib.math_utils import wrap_angle


class PlottingNode(Node):
    def __init__(self, des_angle=1.274, total_msg_cnt=800):
        super().__init__("plot_node")
        
        self.goal = des_angle
        self.plotted = False
        self.pub = self.create_publisher(TurtleBotState, "/cmd_pose", 10)
        self.pub_timer = self.create_timer(0.25, self.goal_pub_cb)

        self.plot_sub = self.create_subscription(
            TurtleBotState, "/state", self.plot_cb, 10
        )
        print("Commanding a goal angle, and saving a plot!")
        print("Please wait...")

        self.angles = []
        self.total_msg_cnt = total_msg_cnt

    def goal_pub_cb(self):
        if not self.plotted:
            msg = TurtleBotState()
            msg.theta = self.goal
            self.pub.publish(msg)


    def plot_cb(self, msg: TurtleBotState):
        """
        Record data until it's been longer than the set time.
        """
        if len(self.angles) < self.total_msg_cnt:
            self.angles.append(wrap_angle(msg.theta))
        else:
            if not self.plotted:
                self.plot()
                self.plotted = True


    def plot(self):
        num_msgs = len(self.angles)
        theta_traj = np.array(self.angles)
        time = np.linspace(0, self.total_msg_cnt / 100, num=num_msgs)
        goal_thetas = self.goal * np.ones_like(time)

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(time, theta_traj, linewidth=2, label="theta", c="cornflowerblue")
        ax.plot(time, goal_thetas, linewidth=2, label="goal", c="orange")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Theta")
        fig.legend()
        filename = Path("src/autonomy_repo/plots/p3_output.png")
        try:
            fig.savefig(filename)  # save the figure to file
        except OSError as e:
            print(
                f"Tried to plot to {filename.absolute()}, but directory does not exist!"
            )
            print("Could not save plot, make sure to launch from ~/autonomy_ws!")
        else:
            print(f"Successfully plotted to {filename.absolute()}")
            print("Close this node with CTRL+C!")
        plt.close(fig)


if __name__ == "__main__":
    rclpy.init()
    node = PlottingNode()
    rclpy.spin(node)
    rclpy.shutdown()
