#!/usr/bin/env python3

import typing as T

import numpy as np
import rclpy
from rclpy.parameter import Parameter
from scipy.interpolate import splev, splrep

from asl_tb3_lib.grids import StochOccupancyGrid2D
from asl_tb3_lib.navigation import BaseNavigator, TrajectoryPlan
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState


class Navigator(BaseNavigator):

    def __init__(self) -> None:
        super().__init__("navigator")
        self.kp = 2.0
        self.kpx = 1.0
        self.kpy = 1.0
        self.kdx = 1.0
        self.kdy = 1.0

        self.V_PREV_THRES = 0.0001
        self.v_desired = 0.2

        self.V_prev = 0.
        self.om_prev = 0.
        self.t_prev = 0.

    def compute_heading_control(
        self,
        state: TurtleBotState,
        goal: TurtleBotState,
    ) -> TurtleBotControl:
        err = wrap_angle(goal.theta - state.theta)
        return TurtleBotControl(omega=(err * self.kp))
    
    def compute_trajectory_tracking_control(self,
        state: TurtleBotState,
        plan: TrajectoryPlan,
        t: float,
    ) -> TurtleBotControl:
        """ Compute control target using a trajectory tracking controller

        Args:
            state (TurtleBotState): current robot state
            plan (TrajectoryPlan): planned trajectory
            t (float): current timestep

        Returns:
            TurtleBotControl: control command
        """
        dt = t - self.t_prev  # Compute time step

        # Using spline interpolation, calculated desired x, x_d, and x_dd
        x_d = splev(t, plan.path_x_spline, der=0)
        xd_d = splev(t, plan.path_x_spline, der=1)
        xdd_d = splev(t, plan.path_x_spline, der=2)

        # Using spline interpolation, calculated desired y, y_d, and y_dd
        y_d = splev(t, plan.path_y_spline, der=0)
        yd_d = splev(t, plan.path_y_spline, der=1)
        ydd_d = splev(t, plan.path_y_spline, der=2)

        
        # Calculate u1 and u2 using the virtual control law for trajectory tracking
        u1 = xdd_d + self.kpx*(x_d - state.x) + self.kdx*(xd_d - self.V_prev*np.cos(state.theta))
        u2 = ydd_d + self.kpy*(y_d - state.y) + self.kdy*(yd_d - self.V_prev*np.sin(state.theta))
        
        # Calculating V using the velocity kinematic equation v = v0 + a*dt
        V = self.V_prev + (u1*np.cos(state.theta) + u2*np.sin(state.theta)) * dt
        
        # Calculate omega but being mindful of when V is 0
        if abs(self.V_prev) < self.V_PREV_THRES:  # If the velocity not between the +/- V_PREV_THRES
            self.V_prev = self.V_PREV_THRES

        om = (u2*np.cos(state.theta) - u1*np.sin(state.theta)) / self.V_prev

        # save the commands that were applied and the time
        self.t_prev = t
        self.V_prev = V
        self.om_prev = om
        
        # Create the control message and add the velocity and angular velocity controls
        control_msg = TurtleBotControl()
        control_msg.v = V
        control_msg.omega = om

        return control_msg  # Return control message

    def compute_trajectory_plan(self,
        state: TurtleBotState,
        goal: TurtleBotState,
        occupancy: StochOccupancyGrid2D,
        resolution: float,
        horizon: float,
        v_desired: float = 0.15,
        spline_alpha: float = 0.05,
    ) -> T.Optional[TrajectoryPlan]:
        
        astar = AStar(
            (state.x - horizon, state.y - horizon),
            (state.x + horizon, state.y + horizon),
            (state.x, state.y),
            (goal.x, goal.y),
            occupancy,
            resolution=resolution,
        )

        # Reset previous velocity and time class variables
        self.V_prev = 0
        self.t_prev = 0

        if not astar.solve():
            return None
        
        path = np.asarray(astar.path)

        if len(path) < 4:
            return None

        ts_arr = [0]
        
        distances = np.linalg.norm(np.diff(path, axis=0), axis=1) # Find the distances between each state
        
        time_arr = distances / self.v_desired # Used s = vt to find the time stamps between each state given a constant velocity
        
        # Added each element in time_arr to properly define the time stamps at each state
        for i in range(1, len(time_arr) + 1):
            ts_arr.append(ts_arr[-1] + time_arr[i - 1])
            
        ts = np.array(ts_arr) # Converted time stamp array to a numpy array
        
        # Uses splrep to fit the cubic splines for each x and y coordinate with respect to the time stamp array
        path_x_spline = splrep(ts, path[:, 0], s=spline_alpha)
        path_y_spline = splrep(ts, path[:, 1], s=spline_alpha)
        
        ###### YOUR CODE END HERE ######
        
        return TrajectoryPlan(
            path=path,
            path_x_spline=path_x_spline,
            path_y_spline=path_y_spline,
            duration=ts[-1],
        )


class AStar(object):
    """Represents a motion planning problem to be solved using A*"""

    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, occupancy, resolution=1):
        self.statespace_lo = statespace_lo         # state space lower bound (e.g., [-5, -5])
        self.statespace_hi = statespace_hi         # state space upper bound (e.g., [5, 5])
        self.occupancy = occupancy                 # occupancy grid (a DetOccupancyGrid2D object)
        self.resolution = resolution               # resolution of the discretization of state space (cell/m)
        self.x_offset = x_init                     
        self.x_init = self.snap_to_grid(x_init)    # initial state
        self.x_goal = self.snap_to_grid(x_goal)    # goal state

        self.closed_set = set()    # the set containing the states that have been visited
        self.open_set = set()      # the set containing the states that are condidate for future expension

        self.est_cost_through = {}  # dictionary of the estimated cost from start to goal passing through state (often called f score)
        self.cost_to_arrive = {}    # dictionary of the cost-to-arrive at state from start (often called g score)
        self.came_from = {}         # dictionary keeping track of each state's parent to reconstruct the path

        self.open_set.add(self.x_init)
        self.cost_to_arrive[self.x_init] = 0
        self.est_cost_through[self.x_init] = self.distance(self.x_init,self.x_goal)

        self.path = None        # the final path as a list of states

    def is_free(self, x):
        """
        Checks if a give state x is free, meaning it is inside the bounds of the map and
        is not inside any obstacle.
        Inputs:
            x: state tuple
        Output:
            Boolean True/False
        Hint: self.occupancy is a DetOccupancyGrid2D object, take a look at its methods for what might be
              useful here
        """
        return (
            self.occupancy.is_free(np.array(x))
            and self.statespace_lo[0] <= x[0] <= self.statespace_hi[0]
            and self.statespace_lo[1] <= x[1] <= self.statespace_hi[1]
        )

    def distance(self, x1, x2):
        """
        Computes the Euclidean distance between two states.
        Inputs:
            x1: First state tuple
            x2: Second state tuple
        Output:
            Float Euclidean distance

        HINT: This should take one line. Tuples can be converted to numpy arrays using np.array().
        """
        return np.linalg.norm(np.array(x2) - np.array(x1))

    def snap_to_grid(self, x):
        """ Returns the closest point on a discrete state grid
        Input:
            x: tuple state
        Output:
            A tuple that represents the closest point to x on the discrete state grid
        """
        return (
            self.resolution * round((x[0] - self.x_offset[0]) / self.resolution) + self.x_offset[0],
            self.resolution * round((x[1] - self.x_offset[1]) / self.resolution) + self.x_offset[1],
        )

    def get_neighbors(self, x):
        """
        Gets the FREE neighbor states of a given state x. Assumes a motion model
        where we can move up, down, left, right, or along the diagonals by an
        amount equal to self.resolution.
        Input:
            x: tuple state
        Ouput:
            List of neighbors that are free, as a list of TUPLES

        HINTS: Use self.is_free to check whether a given state is indeed free.
               Use self.snap_to_grid (see above) to ensure that the neighbors
               you compute are actually on the discrete grid, i.e., if you were
               to compute neighbors by adding/subtracting self.resolution from x,
               numerical errors could creep in over the course of many additions
               and cause grid point equality checks to fail. To remedy this, you
               should make sure that every neighbor is snapped to the grid as it
               is computed.
        """
        neighbors = []
        for dx1 in [-self.resolution, 0, self.resolution]:
            for dx2 in [-self.resolution, 0, self.resolution]:
                x_prime = self.snap_to_grid(x + np.array([dx1, dx2]))
                if self.is_free(x_prime):
                    neighbors.append(x_prime)
        return neighbors

    def find_best_est_cost_through(self):
        """
        Gets the state in open_set that has the lowest est_cost_through
        Output: A tuple, the state found in open_set that has the lowest est_cost_through
        """
        return min(self.open_set, key=lambda x: self.est_cost_through[x])

    def reconstruct_path(self):
        """
        Use the came_from map to reconstruct a path from the initial location to
        the goal location
        Output:
            A list of tuples, which is a list of the states that go from start to goal
        """
        path = [self.x_goal]
        current = path[-1]
        while current != self.x_init:
            path.append(self.came_from[current])
            current = path[-1]
        return list(reversed(path))

    def solve(self):
        """
        Solves the planning problem using the A* search algorithm. It places
        the solution as a list of tuples (each representing a state) that go
        from self.x_init to self.x_goal inside the variable self.path
        Input:
            None
        Output:
            Boolean, True if a solution from x_init to x_goal was found

        HINTS:  We're representing the open and closed sets using python's built-in
                set() class. This allows easily adding and removing items using
                .add(item) and .remove(item) respectively, as well as checking for
                set membership efficiently using the syntax "if item in set".
        """
        while len(self.open_set) > 0:
            x_current = self.find_best_est_cost_through()
            if x_current == self.x_goal:
                self.path = self.reconstruct_path()
                return True
            self.open_set.remove(x_current)
            self.closed_set.add(x_current)
            for x_neigh in self.get_neighbors(x_current):
                if x_neigh in self.closed_set:
                    continue
                tentative_cost_to_arrive = self.cost_to_arrive[x_current] + self.distance(x_current, x_neigh)
                if x_neigh not in self.open_set:
                    self.open_set.add(x_neigh)
                elif tentative_cost_to_arrive > self.cost_to_arrive[x_neigh]:
                    continue
                self.came_from[x_neigh] = x_current
                self.cost_to_arrive[x_neigh] = tentative_cost_to_arrive
                self.est_cost_through[x_neigh] = tentative_cost_to_arrive + self.distance(x_neigh, self.x_goal)
        return False


if __name__ == "__main__":
    rclpy.init()
    node = Navigator()
    rclpy.spin(node)
    rclpy.shutdown()
