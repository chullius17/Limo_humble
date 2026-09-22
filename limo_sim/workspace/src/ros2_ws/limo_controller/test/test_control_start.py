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

"""Verify explicit path execution through the control GUI service."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip('rclpy')
from geometry_msgs.msg import PoseStamped  # noqa: E402
from nav_msgs.msg import Path  # noqa: E402
import rclpy  # noqa: E402
from rclpy.task import Future  # noqa: E402
from std_srvs.srv import SetBool  # noqa: E402

from limo_controller.path_executor import PathExecutor  # noqa: E402


def test_start_control_sends_the_stored_path():
    path = Path()
    path.poses.append(PoseStamped())
    sent = []
    states = []
    bridge = SimpleNamespace(
        goal_pending=False,
        waiting_for_server=False,
        control_path=path,
        active_control_goal_handle=None,
        follow_path_client=SimpleNamespace(
            wait_for_server=lambda timeout_sec: True),
        control_requested=False,
        control_paused=False,
        _send_control_goal=lambda: sent.append(path) or True,
        _publish_control_state=states.append,
    )
    request = SimpleNamespace(data=True)
    response = SimpleNamespace(success=False, message='')

    result = PathExecutor._set_control_active(
        bridge, request, response)

    assert result.success is True
    assert result.message == 'Control start requested.'
    assert bridge.control_requested is True
    assert bridge.control_paused is False
    assert sent == [path]


def test_execution_path_uses_latest_tf_without_mutating_stored_path():
    path = Path()
    path.header.stamp.sec = 10
    pose = PoseStamped()
    pose.header.stamp.sec = 10
    path.poses.append(pose)

    execution = PathExecutor._path_for_execution(path)

    assert execution is not path
    assert execution.header.stamp.sec == 0
    assert execution.header.stamp.nanosec == 0
    assert execution.poses[0].header.stamp.sec == 0
    assert execution.poses[0].header.stamp.nanosec == 0
    assert path.header.stamp.sec == 10
    assert path.poses[0].header.stamp.sec == 10


def test_follow_path_goal_supports_foxy_action_definition():
    path = Path()

    goal = PathExecutor._build_follow_path_goal(
        path,
        'FollowPath',
        'goal_checker',
    )

    assert goal.path == path
    assert goal.controller_id == 'FollowPath'
    if hasattr(goal, 'goal_checker_id'):
        assert goal.goal_checker_id == 'goal_checker'


@pytest.fixture
def executor():
    rclpy.init()
    node = PathExecutor()
    node.follow_path_client.wait_for_server = Mock(return_value=True)
    node.follow_path_client.send_goal_async = Mock(return_value=Future())
    node.control_status_publisher = Mock()
    yield node
    node.destroy_node()
    rclpy.shutdown()


def ready_path(executor):
    path = Path()
    path.header.frame_id = 'map'
    path.header.stamp.sec = 42
    path.poses = [PoseStamped()]
    executor._path_callback(path)
    return path


def start(executor):
    return executor._set_control_active(
        SetBool.Request(data=True), SetBool.Response())


def accept_pending_goal(executor):
    handle = Mock(accepted=True)
    handle.get_result_async.return_value = Future()
    executor.follow_path_client.send_goal_async.return_value.set_result(handle)
    return handle


def test_receiving_a_path_waits_for_start_and_sends_only_once(executor):
    path = ready_path(executor)
    executor.follow_path_client.send_goal_async.assert_not_called()
    assert start(executor).success
    assert start(executor).success
    executor.follow_path_client.send_goal_async.assert_called_once()
    sent = executor.follow_path_client.send_goal_async.call_args[0][0].path
    assert len(sent.poses) == 1
    assert sent.header.frame_id == 'map'
    assert sent.header.stamp.sec == 0
    assert path.header.stamp.sec == 42
    assert executor.control_path.header.stamp.sec == 42


@pytest.mark.parametrize('operation', ['abort', 'pause', 'invalidate'])
def test_stop_before_acceptance_cancels_the_pending_goal(executor, operation):
    ready_path(executor)
    assert start(executor).success
    if operation == 'abort':
        executor._set_control_active(SetBool.Request(data=False), SetBool.Response())
    elif operation == 'pause':
        executor._set_control_enabled(SetBool.Request(data=False), SetBool.Response())
    else:
        executor._path_callback(Path())
    handle = accept_pending_goal(executor)
    handle.cancel_goal_async.assert_called_once()
    assert not executor.control_status_publisher.publish.call_args[0][0].data.startswith(
        'ACTIVE:')


def test_replacing_an_active_path_cancels_and_requires_another_start(executor):
    from action_msgs.msg import GoalStatus
    ready_path(executor)
    assert start(executor).success
    handle = accept_pending_goal(executor)
    ready_path(executor)
    handle.cancel_goal_async.assert_called_once()
    assert not executor.control_requested
    assert not start(executor).success  # Old action has not finished canceling.
    handle.get_result_async.return_value.set_result(
        SimpleNamespace(status=GoalStatus.STATUS_CANCELED))
    executor.follow_path_client.send_goal_async.assert_called_once()
    executor.follow_path_client.send_goal_async.return_value = Future()
    assert start(executor).success
    assert executor.follow_path_client.send_goal_async.call_count == 2


def test_unavailable_controller_queues_start_until_server_is_ready(executor):
    ready_path(executor)
    executor.follow_path_client.wait_for_server.return_value = False
    assert start(executor).success
    assert executor.control_requested
    assert executor.waiting_for_server
    executor.follow_path_client.send_goal_async.assert_not_called()

    executor.follow_path_client.wait_for_server.return_value = True
    executor._retry_waiting_control_goal()

    assert not executor.waiting_for_server
    assert executor.goal_pending
    executor.follow_path_client.send_goal_async.assert_called_once()


def test_abort_removes_a_queued_start(executor):
    ready_path(executor)
    executor.follow_path_client.wait_for_server.return_value = False
    assert start(executor).success

    executor._set_control_active(
        SetBool.Request(data=False), SetBool.Response())
    executor.follow_path_client.wait_for_server.return_value = True
    executor._retry_waiting_control_goal()

    assert not executor.waiting_for_server
    assert not executor.control_requested
    executor.follow_path_client.send_goal_async.assert_not_called()


def test_send_exception_returns_failure_without_killing_services(executor):
    ready_path(executor)
    executor.follow_path_client.send_goal_async.side_effect = RuntimeError('transport failed')
    assert not start(executor).success
    assert not executor.goal_pending
    assert not executor.control_requested
