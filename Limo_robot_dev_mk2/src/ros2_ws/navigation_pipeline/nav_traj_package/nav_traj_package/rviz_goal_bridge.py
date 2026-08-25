# Copyright 2026 Giulio Cataldo
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Forward RViz goal poses to the Nav2 path-planning action."""

import math

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputePathToPose
import rclpy
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


class RvizGoalBridge(Node):
    """Convert RViz ``/goal_pose`` messages into SMAC planning requests."""

    def __init__(self):
        super().__init__('rviz_goal_bridge')

        self.declare_parameter('goal_topic', '/goal_pose')
        self.declare_parameter(
            'compute_path_action',
            '/compute_path_to_pose',
        )
        self.declare_parameter('planner_id', 'GridBased')

        goal_topic = str(self.get_parameter('goal_topic').value)
        action_name = str(
            self.get_parameter('compute_path_action').value
        )
        self.planner_id = str(self.get_parameter('planner_id').value)

        # This bridge intentionally keeps goal selection graphical: users can
        # choose both position and final heading with RViz's 2D Goal Pose tool
        # instead of manually composing a ComputePathToPose action request.
        self.goal_subscription = self.create_subscription(
            PoseStamped,
            goal_topic,
            self._goal_callback,
            10,
        )
        self.compute_path_client = ActionClient(
            self,
            ComputePathToPose,
            action_name,
        )
        self.active_goal_handle = None

        self.get_logger().info(
            'RViz goal bridge started to simplify graphical goal selection: '
            f'{goal_topic} -> {action_name} '
            f'(planner={self.planner_id})'
        )

    def _goal_callback(self, pose: PoseStamped) -> None:
        """Request a path to the pose selected with RViz."""
        if not pose.header.frame_id:
            self.get_logger().error(
                'Ignoring RViz goal without a frame_id.'
            )
            return

        if not self.compute_path_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().error(
                'The ComputePathToPose action server is not available.'
            )
            return

        if self.active_goal_handle is not None:
            self.active_goal_handle.cancel_goal_async()
            self.active_goal_handle = None

        request = ComputePathToPose.Goal()
        request.goal = pose
        request.planner_id = self.planner_id
        request.use_start = False

        yaw = self._quaternion_to_yaw(pose)
        self.get_logger().info(
            'Requesting path to RViz goal: '
            f'frame={pose.header.frame_id}, '
            f'x={pose.pose.position.x:.3f}, '
            f'y={pose.pose.position.y:.3f}, '
            f'yaw={yaw:.3f} rad'
        )
        future = self.compute_path_client.send_goal_async(request)
        future.add_done_callback(self._goal_response_callback)

    def _goal_response_callback(self, future) -> None:
        """Start monitoring an accepted planning request."""
        try:
            goal_handle = future.result()
        except Exception as exc:  # noqa: B902
            self.get_logger().error(
                f'Failed to send the planning request: {exc}'
            )
            return

        if not goal_handle.accepted:
            self.get_logger().warning('The planning goal was rejected.')
            return

        self.active_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda result, handle=goal_handle: self._result_callback(
                result,
                handle,
            )
        )

    def _result_callback(self, future, goal_handle) -> None:
        """Report whether SMAC produced a path for the selected pose."""
        if self.active_goal_handle is goal_handle:
            self.active_goal_handle = None

        try:
            response = future.result()
        except Exception as exc:  # noqa: B902
            self.get_logger().error(
                f'Failed to receive the planning result: {exc}'
            )
            return

        if response.status == GoalStatus.STATUS_SUCCEEDED:
            pose_count = len(response.result.path.poses)
            self.get_logger().info(
                f'RViz goal planned successfully: {pose_count} path poses.'
            )
        elif response.status == GoalStatus.STATUS_CANCELED:
            self.get_logger().info(
                'Previous RViz planning goal canceled.'
            )
        else:
            self.get_logger().warning(
                'SMAC could not find a valid path to the RViz goal '
                f'(action status={response.status}).'
            )

    @staticmethod
    def _quaternion_to_yaw(pose: PoseStamped) -> float:
        """Return the planar yaw encoded by a PoseStamped quaternion."""
        quaternion = pose.pose.orientation
        return math.atan2(
            2.0 * (
                quaternion.w * quaternion.z
                + quaternion.x * quaternion.y
            ),
            1.0 - 2.0 * (
                quaternion.y * quaternion.y
                + quaternion.z * quaternion.z
            ),
        )


def main(args=None):
    """Run the RViz-to-planner goal bridge."""
    rclpy.init(args=args)
    node = RvizGoalBridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
