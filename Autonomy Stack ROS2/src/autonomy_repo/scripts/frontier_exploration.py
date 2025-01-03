#!/usr/bin/env python3

from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState
from asl_tb3_lib.grids import StochOccupancyGrid2D
from nav_msgs.msg import OccupancyGrid
import rclpy
import numpy as np
from std_msgs.msg import Bool
import typing as T

from scipy.signal import convolve2d
from rclpy.node import Node

class FrontierExploration(Node):
    def __init__(self):
        super().__init__("frontier_exploration")
        
        self.get_logger().info("Launched frontier exploration")

        self.occupancy: T.Optional[StochOccupancyGrid2D] = None
        self.state: T.Optional[TurtleBotState] = None
        
        self.explore_init = True
        self.exploring = True
        self.finished_exploring = False
        
        # Detector variables
        self.active = True
        self.prev_time = None
        
        self.cmd_nav_pub = self.create_publisher(TurtleBotState, "/cmd_nav", 10)

        self.detector_bool_sub = self.create_subscription(Bool, '/detector_bool', self.detector_bool_callback, 10)
        self.map_sub = self.create_subscription(OccupancyGrid, "/map", self.map_callback, 10)
        self.nav_success_sub = self.create_subscription(Bool, "/nav_success", self.nav_success_callback, 10)
        self.state_sub = self.create_subscription(TurtleBotState, "/state", self.state_callback, 10)
        
    def detector_bool_callback(self, msg: Bool):
        curr_time = self.get_clock().now().nanoseconds / 1e9
        if msg.data and self.state is not None:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
            if self.prev_time is None:
                self.active = False
                self.prev_time = self.get_clock().now().nanoseconds / 1e9
                self.get_logger().info("Stopping")
                self.cmd_nav_pub.publish(TurtleBotState(x=self.state.x, y=self.state.y, theta=self.state.theta))
        if self.prev_time is not None:
            dt = curr_time - self.prev_time
            if dt >= 5 and not self.active:
                self.active = True
                self.get_logger().info("Resuming")
                self.nav_success_callback(None)
            if dt >= 8:  # wait a bit
                self.prev_time = None

    def map_callback(self, msg: OccupancyGrid) -> None:
        """ Callback triggered when the map is updated

        Args:
            msg (OccupancyGrid): updated map message
        """
        self.occupancy = StochOccupancyGrid2D(
            resolution=msg.info.resolution,
            size_xy=np.array([msg.info.width, msg.info.height]),
            origin_xy=np.array([msg.info.origin.position.x, msg.info.origin.position.y]),
            window_size=7,
            probs=msg.data,
        )
            
    def state_callback(self, msg: TurtleBotState) -> None:
        if self.explore_init and not self.occupancy is None:
            self.state = msg
            self.explore_init = False
            self.nav_success_callback(None)
        else:
            self.state = msg        

    def nav_success_callback(self, msg):
        """ returns potential states to explore
        Args:
            occupancy (StochasticOccupancyGrid2D): Represents the known, unknown, occupied, and unoccupied states. See class in first section of notebook.

        Returns:
            frontier_states (np.ndarray): state-vectors in (x, y) coordinates of potential states to explore. Shape is (N, 2), where N is the number of possible states to explore.

        HINTS:
        - Function `convolve2d` may be helpful in producing the number of unknown, and number of occupied states in a window of a specified cell
        - Note the distinction between physical states and grid cells. Most operations can be done on grid cells, and converted to physical states at the end of the function with `occupancy.grid2state()`
        """

        if not self.active or self.occupancy is None or self.state is None:
            return
        
        self.get_logger().info("Currently Exploring")

        window_size = 13

        occupied = convolve2d(
            (self.occupancy.probs > 0.5),
            np.ones((window_size, window_size)),
            mode="same"
        ).T
        unoccupied = convolve2d(
            (self.occupancy.probs == 0),
            np.ones((window_size, window_size)),
            mode="same"
        ).T
        unknown = convolve2d(
            (self.occupancy.probs == -1),
            np.ones((window_size, window_size)),
            mode="same"
        ).T
        frontier_states = self.occupancy.grid2state(
            np.array(
                np.nonzero(
                    (occupied == 0)
                    & (unoccupied > 0.3 * window_size ** 2)
                    & (unknown > 0.2 * window_size ** 2)
                )
            ).T
        )

        if len(frontier_states) == 0:
            self.exploring = False
            self.finished_exploring = True
            self.get_logger().info("Finished exploration")
            return

        distances = np.linalg.norm(
            frontier_states - np.array([self.state.x, self.state.y]),
            axis=1
        )
        nearest_state = frontier_states[np.argmin(distances)]

        goal_state = TurtleBotState()
        goal_state.x = float(nearest_state[0])
        goal_state.y = float(nearest_state[1])
        goal_state.theta = 0.0
        self.get_logger().info(f"Goal state: {goal_state.x, goal_state.y}")
        self.cmd_nav_pub.publish(goal_state)

# Define main block for node execution
if __name__ == "__main__":
    rclpy.init()
    node = FrontierExploration()
    rclpy.spin(node)
    rclpy.shutdown()
