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

"""Forward RViz goals to Nav2 and recover nearby feasible goal poses."""

import copy
from dataclasses import dataclass
import math

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputePathToPose
from nav_msgs.msg import OccupancyGrid
import rclpy
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy


@dataclass
class GoalCandidate:
    """Represent one nearby pose considered by the recovery search."""

    pose: PoseStamped
    position_delta: float
    angle_delta: float
    score: float


class RvizGoalBridge(Node):
    """Convert RViz ``/goal_pose`` messages into robust SMAC requests."""

    def __init__(self):
        super().__init__('rviz_goal_bridge')

        self.declare_parameter('goal_topic', '/goal_pose')
        self.declare_parameter(
            'compute_path_action',
            '/compute_path_to_pose',
        )
        self.declare_parameter('planner_id', 'GridBased')
        self.declare_parameter(
            'costmap_topic',
            '/global_costmap/costmap',
        )
        self.declare_parameter(
            'adjusted_goal_topic',
            '/adjusted_goal_pose',
        )
        self.declare_parameter('enable_goal_adjustment', True)
        self.declare_parameter('position_search_radius', 0.75)
        self.declare_parameter('position_search_step', 0.10)
        self.declare_parameter('angle_search_step_deg', 22.5)
        self.declare_parameter('orientation_score_weight', 0.10)
        self.declare_parameter('max_planning_attempts', 48)
        self.declare_parameter('footprint_length', 0.32)
        self.declare_parameter('footprint_width', 0.20)
        self.declare_parameter('collision_cost_threshold', 99)

        goal_topic = str(self.get_parameter('goal_topic').value)
        action_name = str(
            self.get_parameter('compute_path_action').value
        )
        costmap_topic = str(self.get_parameter('costmap_topic').value)
        adjusted_goal_topic = str(
            self.get_parameter('adjusted_goal_topic').value
        )
        self.planner_id = str(self.get_parameter('planner_id').value)
        self.enable_goal_adjustment = bool(
            self.get_parameter('enable_goal_adjustment').value
        )
        self.position_search_radius = float(
            self.get_parameter('position_search_radius').value
        )
        self.position_search_step = float(
            self.get_parameter('position_search_step').value
        )
        self.angle_search_step = math.radians(float(
            self.get_parameter('angle_search_step_deg').value
        ))
        self.orientation_score_weight = float(
            self.get_parameter('orientation_score_weight').value
        )
        self.max_planning_attempts = int(
            self.get_parameter('max_planning_attempts').value
        )
        self.footprint_half_length = 0.5 * float(
            self.get_parameter('footprint_length').value
        )
        self.footprint_half_width = 0.5 * float(
            self.get_parameter('footprint_width').value
        )
        self.collision_cost_threshold = int(
            self.get_parameter('collision_cost_threshold').value
        )
        self._validate_parameters()

        # This bridge intentionally keeps goal selection graphical: users can
        # choose both position and final heading with RViz's 2D Goal Pose tool
        # instead of manually composing a ComputePathToPose action request.
        self.goal_subscription = self.create_subscription(
            PoseStamped,
            goal_topic,
            self._goal_callback,
            10,
        )
        costmap_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.costmap_subscription = self.create_subscription(
            OccupancyGrid,
            costmap_topic,
            self._costmap_callback,
            costmap_qos,
        )
        self.adjusted_goal_publisher = self.create_publisher(
            PoseStamped,
            adjusted_goal_topic,
            10,
        )
        self.compute_path_client = ActionClient(
            self,
            ComputePathToPose,
            action_name,
        )

        self.costmap = None
        self.active_goal_handle = None
        self.search_generation = 0
        self.candidates = []
        self.candidate_index = 0
        self.planning_attempts = 0

        self.get_logger().info(
            'RViz goal bridge started to simplify graphical goal selection: '
            f'{goal_topic} -> {action_name} '
            f'(planner={self.planner_id}, '
            f'adjustment={self.enable_goal_adjustment}, '
            f'radius={self.position_search_radius:.2f} m, '
            f'max_attempts={self.max_planning_attempts})'
        )

    def _validate_parameters(self) -> None:
        """Reject recovery settings that could produce invalid searches."""
        if self.position_search_radius < 0.0:
            raise ValueError('position_search_radius must be non-negative')
        if self.position_search_step <= 0.0:
            raise ValueError('position_search_step must be positive')
        if not 0.0 < self.angle_search_step <= math.pi:
            raise ValueError('angle_search_step_deg must be in (0, 180]')
        if self.orientation_score_weight < 0.0:
            raise ValueError('orientation_score_weight must be non-negative')
        if self.max_planning_attempts <= 0:
            raise ValueError('max_planning_attempts must be positive')
        if self.footprint_half_length <= 0.0:
            raise ValueError('footprint_length must be positive')
        if self.footprint_half_width <= 0.0:
            raise ValueError('footprint_width must be positive')
        if not 0 <= self.collision_cost_threshold <= 100:
            raise ValueError('collision_cost_threshold must be in [0, 100]')

    def _costmap_callback(self, msg: OccupancyGrid) -> None:
        """Cache the latest global costmap used by SMAC."""
        expected_size = msg.info.width * msg.info.height
        if len(msg.data) == expected_size:
            self.costmap = msg

    def _goal_callback(self, pose: PoseStamped) -> None:
        """Start a bounded feasibility search for an RViz-selected pose."""
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

        self.search_generation += 1
        generation = self.search_generation
        if self.active_goal_handle is not None:
            self.active_goal_handle.cancel_goal_async()
            self.active_goal_handle = None

        candidates = self._build_candidates(pose)
        if self.costmap is not None:
            candidates = [
                candidate for candidate in candidates
                if self._is_footprint_free(candidate.pose)
            ]
        else:
            self.get_logger().warning(
                'Global costmap not received yet; candidate collision '
                'filtering is unavailable.'
            )

        self.candidates = candidates
        self.candidate_index = 0
        self.planning_attempts = 0

        requested_yaw = self._quaternion_to_yaw(pose.pose.orientation)
        self.get_logger().info(
            'Searching a feasible RViz goal near: '
            f'frame={pose.header.frame_id}, '
            f'x={pose.pose.position.x:.3f}, '
            f'y={pose.pose.position.y:.3f}, '
            f'yaw={requested_yaw:.3f} rad; '
            f'collision-free candidates={len(candidates)}'
        )
        self._send_next_candidate(generation)

    def _build_candidates(self, requested: PoseStamped):
        """Create nearby poses ordered by position and heading changes."""
        if not self.enable_goal_adjustment:
            return [GoalCandidate(copy.deepcopy(requested), 0.0, 0.0, 0.0)]

        positions = [(0.0, 0.0, 0.0)]
        ring_count = int(
            self.position_search_radius / self.position_search_step
            + 1.0e-9
        )
        for ring in range(1, ring_count + 1):
            radius = ring * self.position_search_step
            sample_count = max(
                8,
                int(math.ceil(
                    2.0 * math.pi * radius
                    / self.position_search_step
                )),
            )
            for sample in range(sample_count):
                bearing = 2.0 * math.pi * sample / sample_count
                positions.append((
                    radius,
                    radius * math.cos(bearing),
                    radius * math.sin(bearing),
                ))

        angle_offsets = [0.0]
        angle_steps = int(math.ceil(math.pi / self.angle_search_step))
        for step in range(1, angle_steps + 1):
            offset = min(step * self.angle_search_step, math.pi)
            angle_offsets.append(offset)
            if offset < math.pi - 1.0e-9:
                angle_offsets.append(-offset)

        requested_yaw = self._quaternion_to_yaw(
            requested.pose.orientation
        )
        candidates = []
        for radius, delta_x, delta_y in positions:
            for angle_delta in angle_offsets:
                candidate = copy.deepcopy(requested)
                candidate.pose.position.x += delta_x
                candidate.pose.position.y += delta_y
                self._set_pose_yaw(candidate, requested_yaw + angle_delta)
                score = (
                    radius
                    + self.orientation_score_weight * abs(angle_delta)
                )
                candidates.append(GoalCandidate(
                    pose=candidate,
                    position_delta=radius,
                    angle_delta=angle_delta,
                    score=score,
                ))

        candidates.sort(key=lambda candidate: (
            candidate.score,
            candidate.position_delta,
            abs(candidate.angle_delta),
        ))
        return candidates

    def _is_footprint_free(self, pose: PoseStamped) -> bool:
        """Check every costmap cell covered by the candidate footprint."""
        grid = self.costmap
        if grid is None:
            return True
        if pose.header.frame_id != grid.header.frame_id:
            return True

        resolution = grid.info.resolution
        if resolution <= 0.0:
            return False
        origin = grid.info.origin
        origin_yaw = self._quaternion_to_yaw(origin.orientation)
        origin_cos = math.cos(origin_yaw)
        origin_sin = math.sin(origin_yaw)
        candidate_x = pose.pose.position.x
        candidate_y = pose.pose.position.y
        candidate_yaw = self._quaternion_to_yaw(pose.pose.orientation)
        candidate_cos = math.cos(candidate_yaw)
        candidate_sin = math.sin(candidate_yaw)
        radius = math.hypot(
            self.footprint_half_length,
            self.footprint_half_width,
        )

        grid_corners = [
            self._world_to_grid_continuous(
                candidate_x + delta_x,
                candidate_y + delta_y,
                origin.position.x,
                origin.position.y,
                origin_cos,
                origin_sin,
                resolution,
            )
            for delta_x in (-radius, radius)
            for delta_y in (-radius, radius)
        ]
        min_x = max(0, math.floor(min(x for x, _ in grid_corners)))
        max_x = min(
            grid.info.width - 1,
            math.floor(max(x for x, _ in grid_corners)),
        )
        min_y = max(0, math.floor(min(y for _, y in grid_corners)))
        max_y = min(
            grid.info.height - 1,
            math.floor(max(y for _, y in grid_corners)),
        )
        if min_x > max_x or min_y > max_y:
            return False

        checked_cell = False
        for grid_y in range(min_y, max_y + 1):
            for grid_x in range(min_x, max_x + 1):
                local_x = (grid_x + 0.5) * resolution
                local_y = (grid_y + 0.5) * resolution
                world_x = (
                    origin.position.x
                    + origin_cos * local_x
                    - origin_sin * local_y
                )
                world_y = (
                    origin.position.y
                    + origin_sin * local_x
                    + origin_cos * local_y
                )
                delta_x = world_x - candidate_x
                delta_y = world_y - candidate_y
                footprint_x = (
                    candidate_cos * delta_x
                    + candidate_sin * delta_y
                )
                footprint_y = (
                    -candidate_sin * delta_x
                    + candidate_cos * delta_y
                )
                if (
                    abs(footprint_x) > self.footprint_half_length
                    or abs(footprint_y) > self.footprint_half_width
                ):
                    continue

                checked_cell = True
                cost = grid.data[grid_y * grid.info.width + grid_x]
                if (
                    cost < 0
                    or cost >= self.collision_cost_threshold
                ):
                    return False
        return checked_cell

    def _send_next_candidate(self, generation: int) -> None:
        """Send candidates sequentially until SMAC accepts one."""
        if generation != self.search_generation:
            return
        if self.planning_attempts >= self.max_planning_attempts:
            self.get_logger().warning(
                'No feasible adjusted goal found after '
                f'{self.planning_attempts} planning attempts.'
            )
            return
        if self.candidate_index >= len(self.candidates):
            self.get_logger().warning(
                'No collision-free goal candidate produced a valid path.'
            )
            return

        candidate = self.candidates[self.candidate_index]
        self.candidate_index += 1
        self.planning_attempts += 1
        candidate.pose.header.stamp = self.get_clock().now().to_msg()

        request = ComputePathToPose.Goal()
        request.goal = candidate.pose
        request.planner_id = self.planner_id
        request.use_start = False
        future = self.compute_path_client.send_goal_async(request)
        future.add_done_callback(
            lambda response, current_generation=generation,
            current_candidate=candidate: self._goal_response_callback(
                response,
                current_generation,
                current_candidate,
            )
        )

    def _goal_response_callback(
        self,
        future,
        generation: int,
        candidate: GoalCandidate,
    ) -> None:
        """Monitor an accepted candidate or continue the recovery search."""
        try:
            goal_handle = future.result()
        except Exception as exc:  # noqa: B902
            if generation == self.search_generation:
                self.get_logger().error(
                    f'Failed to send a planning request: {exc}'
                )
                self._send_next_candidate(generation)
            return

        if generation != self.search_generation:
            if goal_handle.accepted:
                goal_handle.cancel_goal_async()
            return
        if not goal_handle.accepted:
            self._send_next_candidate(generation)
            return

        self.active_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda result, handle=goal_handle,
            current_generation=generation,
            current_candidate=candidate: self._result_callback(
                result,
                handle,
                current_generation,
                current_candidate,
            )
        )

    def _result_callback(
        self,
        future,
        goal_handle,
        generation: int,
        candidate: GoalCandidate,
    ) -> None:
        """Publish the closest successful pose or try the next candidate."""
        if self.active_goal_handle is goal_handle:
            self.active_goal_handle = None
        if generation != self.search_generation:
            return

        try:
            response = future.result()
        except Exception as exc:  # noqa: B902
            self.get_logger().error(
                f'Failed to receive a planning result: {exc}'
            )
            self._send_next_candidate(generation)
            return

        if response.status == GoalStatus.STATUS_SUCCEEDED:
            candidate.pose.header.stamp = self.get_clock().now().to_msg()
            self.adjusted_goal_publisher.publish(candidate.pose)
            pose_count = len(response.result.path.poses)
            self.get_logger().info(
                'RViz goal planned successfully: '
                f'{pose_count} path poses, '
                f'position adjustment={candidate.position_delta:.3f} m, '
                f'angle adjustment={math.degrees(candidate.angle_delta):.1f} '
                f'deg, attempts={self.planning_attempts}.'
            )
            return

        self._send_next_candidate(generation)

    @staticmethod
    def _world_to_grid_continuous(
        world_x: float,
        world_y: float,
        origin_x: float,
        origin_y: float,
        origin_cos: float,
        origin_sin: float,
        resolution: float,
    ):
        """Transform world coordinates into continuous grid coordinates."""
        delta_x = world_x - origin_x
        delta_y = world_y - origin_y
        return (
            (origin_cos * delta_x + origin_sin * delta_y) / resolution,
            (-origin_sin * delta_x + origin_cos * delta_y) / resolution,
        )

    @staticmethod
    def _quaternion_to_yaw(quaternion) -> float:
        """Return the planar yaw encoded by a quaternion message."""
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

    @staticmethod
    def _set_pose_yaw(pose: PoseStamped, yaw: float) -> None:
        """Write a normalized planar quaternion into a goal pose."""
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = math.sin(0.5 * yaw)
        pose.pose.orientation.w = math.cos(0.5 * yaw)


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
