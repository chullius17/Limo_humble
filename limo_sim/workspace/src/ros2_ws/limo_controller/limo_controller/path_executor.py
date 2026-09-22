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

"""Execute published paths through Nav2 on explicit GUI requests."""

import copy

from action_msgs.msg import GoalStatus
from nav2_msgs.action import FollowPath
from nav_msgs.msg import Path
import rclpy
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool


class PathExecutor(Node):
    """Own path execution, cancellation and the GUI control services."""

    def __init__(self):
        super().__init__('path_executor')
        self.declare_parameter('path_topic', '/limo/planning/path')
        self.declare_parameter('follow_path_action', '/follow_path')
        self.declare_parameter('controller_id', 'FollowPath')
        self.declare_parameter('goal_checker_id', 'goal_checker')
        self.controller_id = self.get_parameter('controller_id').value
        self.goal_checker_id = self.get_parameter('goal_checker_id').value
        self.follow_path_client = ActionClient(
            self, FollowPath, self.get_parameter('follow_path_action').value)
        self.control_path = None
        self.path_generation = 0
        self.goal_pending = False
        self.waiting_for_server = False
        self.active_control_goal_handle = None
        self.control_requested = False
        self.control_paused = False
        self.last_control_feedback_ns = None

        state_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.control_status_publisher = self.create_publisher(
            String, '/limo/control/status', state_qos)
        self.control_active_publisher = self.create_publisher(
            Bool, '/limo/control/active', state_qos)
        self.control_paused_publisher = self.create_publisher(
            Bool, '/limo/control/paused', state_qos)
        self.path_subscription = self.create_subscription(
            Path, self.get_parameter('path_topic').value,
            self._path_callback, state_qos)
        self.start_abort_service = self.create_service(
            SetBool, '/limo/control/set_active', self._set_control_active)
        self.pause_resume_service = self.create_service(
            SetBool, '/limo/control/set_enabled', self._set_control_enabled)
        self.follow_path_retry_timer = self.create_timer(
            0.25, self._retry_waiting_control_goal)
        self._publish_available_path()

    def destroy_node(self):
        """Release the controller action client before its node handle on Foxy."""
        self.follow_path_client.destroy()
        return super().destroy_node()

    def _publish_control_state(self, status):
        self.control_status_publisher.publish(String(data=status))
        self.control_active_publisher.publish(Bool(data=self.control_requested))
        self.control_paused_publisher.publish(Bool(data=self.control_paused))

    def _publish_available_path(self):
        if self.control_path is None:
            self._publish_control_state('IDLE: waiting for a planned path')
        else:
            self._publish_control_state(
                f'READY: path contains {len(self.control_path.poses)} poses')

    def _path_callback(self, path):
        # Empty paths invalidate the previous plan. A replacement never starts
        # automatically, even when it arrives during an active control action.
        self.path_generation += 1
        self.control_requested = False
        self.control_paused = False
        self.waiting_for_server = False
        self.control_path = copy.deepcopy(path) if path.poses else None
        if self.active_control_goal_handle is not None:
            self.active_control_goal_handle.cancel_goal_async()
        self._publish_available_path()

    def _set_control_active(self, request, response):
        if not request.data:
            self.control_requested = False
            self.control_paused = False
            self.waiting_for_server = False
            if self.active_control_goal_handle is not None:
                self.active_control_goal_handle.cancel_goal_async()
            if self.goal_pending or self.active_control_goal_handle is not None:
                self._publish_control_state('ABORTING: canceling path control')
            else:
                self._publish_available_path()
            response.success = True
            response.message = 'Control abort requested.'
            return response

        if self.control_path is None:
            response.success = False
            response.message = 'No planned path is ready.'
        elif (self.waiting_for_server or self.goal_pending
              or self.active_control_goal_handle is not None):
            response.success = self.control_requested
            response.message = (
                'Control is already active.' if self.control_requested
                else 'Wait for control abort to finish.')
        else:
            self.control_requested = True
            self.control_paused = False
            response.success = self._send_control_goal()
            response.message = (
                'Control start requested.' if response.success
                else 'Unable to send path to FollowPath; see control status.')
        return response

    def _set_control_enabled(self, request, response):
        if not self.control_requested:
            response.success = False
            response.message = 'Control has not been started.'
            return response
        if request.data:
            self.control_paused = False
            response.success = True
            if not self.goal_pending and self.active_control_goal_handle is None:
                response.success = self._send_control_goal()
            response.message = (
                'Control resume requested.' if response.success
                else 'Unable to resume path control; see control status.')
        else:
            self.control_paused = True
            if self.active_control_goal_handle is not None:
                self.active_control_goal_handle.cancel_goal_async()
            self._publish_control_state('PAUSED: control pause requested')
            response.success = True
            response.message = 'Control pause requested.'
        return response

    def _fail(self, message):
        self.control_requested = False
        self.control_paused = False
        self.waiting_for_server = False
        self.get_logger().error(message)
        self._publish_control_state(f'ERROR: {message}')

    def _retry_waiting_control_goal(self):
        """Send a queued START as soon as the lifecycle action is available."""
        if not self.waiting_for_server:
            return
        if (not self.control_requested or self.control_paused
                or self.control_path is None):
            return
        self._send_control_goal()

    def _send_control_goal(self):
        try:
            if not self.follow_path_client.wait_for_server(timeout_sec=0.0):
                self.waiting_for_server = True
                self._publish_control_state(
                    'WAITING: FollowPath server is not ready; START is queued')
                return True
            self.waiting_for_server = False
            request = self._build_follow_path_goal(
                self._path_for_execution(self.control_path),
                self.controller_id, self.goal_checker_id)
            generation = self.path_generation
            self.last_control_feedback_ns = None
            self.goal_pending = True
            self._publish_control_state('STARTING: sending path to controller')
            future = self.follow_path_client.send_goal_async(
                request,
                feedback_callback=lambda msg: self._control_feedback_callback(
                    msg, generation),
            )
            future.add_done_callback(
                lambda result: self._control_goal_response_callback(
                    result, generation, len(request.path.poses)))
            return True
        except Exception as exc:
            self.goal_pending = False
            self._fail(f'Failed to send FollowPath goal: {exc}')
            return False

    @staticmethod
    def _path_for_execution(path):
        """Copy the stored path using the latest available transforms."""
        execution_path = copy.deepcopy(path)
        latest = Time().to_msg()
        execution_path.header.stamp = latest
        for pose in execution_path.poses:
            pose.header.stamp = latest
        return execution_path

    @staticmethod
    def _build_follow_path_goal(path, controller_id, goal_checker_id):
        """Build a Nav2 goal, including only fields supported by this ROS version."""
        request = FollowPath.Goal()
        request.path = path
        request.controller_id = controller_id
        # Foxy does not expose goal_checker_id in FollowPath.Goal.
        if hasattr(request, 'goal_checker_id'):
            request.goal_checker_id = goal_checker_id
        return request

    def _control_feedback_callback(self, feedback_msg, generation):
        if (generation != self.path_generation or not self.control_requested
                or self.control_paused):
            return
        now_ns = self.get_clock().now().nanoseconds
        if (self.last_control_feedback_ns is not None
                and 0 <= now_ns - self.last_control_feedback_ns < 500_000_000):
            return
        self.last_control_feedback_ns = now_ns
        feedback = feedback_msg.feedback
        self._publish_control_state(
            'ACTIVE: distance={:.2f} m speed={:.2f} m/s'.format(
                feedback.distance_to_goal, feedback.speed))

    def _control_goal_response_callback(self, future, generation, pose_count):
        self.goal_pending = False
        try:
            goal_handle = future.result()
        except Exception as exc:
            if generation == self.path_generation:
                self._fail(f'Failed to send FollowPath goal: {exc}')
            return
        if not goal_handle.accepted:
            if generation == self.path_generation:
                self._fail('Controller rejected the path')
            return

        self.active_control_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda result: self._control_result_callback(
                result, goal_handle, generation))
        # A stop, pause or new plan may arrive before the action is accepted.
        if (generation != self.path_generation or not self.control_requested
                or self.control_paused):
            goal_handle.cancel_goal_async()
            return
        self._publish_control_state(
            f'ACTIVE: following path with {pose_count} poses')

    def _control_result_callback(self, future, goal_handle, generation):
        if self.active_control_goal_handle is goal_handle:
            self.active_control_goal_handle = None
        if generation != self.path_generation:
            return
        try:
            response = future.result()
        except Exception as exc:
            self._fail(f'Failed to receive FollowPath result: {exc}')
            return
        if response.status == GoalStatus.STATUS_SUCCEEDED:
            self.control_requested = False
            self.control_paused = False
            self._publish_available_path()
        elif response.status == GoalStatus.STATUS_CANCELED:
            if self.control_paused:
                self._publish_control_state('PAUSED: control is stopped')
            elif not self.control_requested:
                self._publish_available_path()
            else:
                # Resume may have arrived while pause cancellation was pending.
                self._send_control_goal()
        else:
            self._fail(f'FollowPath failed with status={response.status}')


def main(args=None):
    """Run the path executor independently of the planning package."""
    rclpy.init(args=args)
    node = PathExecutor()
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
