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

"""Exercise planning output without a controller process."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip('rclpy')
from action_msgs.msg import GoalStatus  # noqa: E402
from geometry_msgs.msg import PoseStamped  # noqa: E402
from nav_msgs.msg import Path  # noqa: E402
import rclpy  # noqa: E402

from traj_package.rviz_goal_bridge import GoalCandidate, RvizGoalBridge  # noqa: E402


@pytest.fixture
def bridge():
    rclpy.init()
    node = RvizGoalBridge()
    node.path_publisher = Mock()
    node.planning_status_publisher = Mock()
    yield node
    node.destroy_node()
    rclpy.shutdown()


def test_success_publishes_path_without_controller(bridge):
    path = Path()
    path.poses = [PoseStamped()]
    candidate = GoalCandidate(PoseStamped(), 0.0, 0.0, 0.0)
    future = Mock()
    future.result.return_value = SimpleNamespace(
        status=GoalStatus.STATUS_SUCCEEDED,
        result=SimpleNamespace(path=path),
    )
    bridge._result_callback(future, None, 0, candidate, candidate)
    bridge.path_publisher.publish.assert_called_once_with(path)
    assert bridge.planning_status_publisher.publish.call_args[0][0].data == (
        'READY: path contains 1 poses')


def test_new_goal_invalidates_path_even_when_planner_is_unavailable(bridge):
    bridge.compute_path_client.wait_for_server = Mock(return_value=False)
    old_goal = Mock()
    bridge.active_planning_goal_handle = old_goal
    pose = PoseStamped()
    pose.header.frame_id = 'map'
    bridge._goal_callback(pose)
    assert bridge.path_publisher.publish.call_args[0][0].poses == []
    old_goal.cancel_goal_async.assert_called_once()
    assert bridge.search_generation == 1
    assert bridge.planning_status_publisher.publish.call_args[0][0].data == (
        'ERROR: planner server unavailable')


def test_late_planning_result_cannot_restore_an_invalidated_path(bridge):
    bridge.search_generation = 1
    bridge._result_callback(Mock(), None, 0, None, None)
    bridge.path_publisher.publish.assert_not_called()
