#!/usr/bin/env python3

import os
import sys
import time
import unittest

from launch import LaunchDescription
import launch.actions
import launch_ros.actions
import launch_testing.actions
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare



def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    path_to_test = os.path.dirname(__file__)

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            IncludeLaunchDescription(
                PathJoinSubstitution(
                    [FindPackageShare("asl_tb3_sim"), "launch", "rviz.launch.py"]
                ),
                launch_arguments={
                    "config": PathJoinSubstitution(
                        [
                            FindPackageShare("autonomy_repo"),
                            "rviz",
                            "default.rviz",
                        ]
                    ),
                    "use_sim_time": use_sim_time,
                }.items(),
            ),
            # relay RVIZ goal pose to some other channel
            Node(
                executable="rviz_goal_relay.py",
                package="asl_tb3_lib",
                parameters=[
                    {"output_channel": "/cmd_nav"},
                ],
            ),
            # state publisher for turtlebot
            Node(
                executable="state_publisher.py",
                package="asl_tb3_lib",
            ),
            # student's navigator node
            Node(
                executable="navigator.py",
                package="autonomy_repo",
                parameters=[
                    {"use_sim_time":use_sim_time}
                ],
            ),
            # student's frontier exploration node
            Node(
                executable="frontier_exploration.py",
                package="autonomy_repo",
                parameters=[
                    {"use_sim_time":use_sim_time}
                ],
            ),
            # launch.actions.TimerAction(
            # period=5.0,
            # actions=[
            #     launch_ros.actions.Node(
            #     # executable="frontier_exploration.py",
            #     executable=sys.executable,
            #     arguments=["src/aa-274a-section/autonomy_ws/src/autonomy_repo/scripts/frontier_exploration.py"],
            #     additional_env={"PYTHONUNBUFFERED": "1"},
            #     name="frontier_exploration_node"
            #     # package="autonomy_repo",
            #     # parameters=[
            #     #     {"use_sim_time":use_sim_time}
            #     # ],
            # )]),
        ]
    )
