#!/usr/bin/env python3

import math
import sys
import threading
from typing import List

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from std_msgs.msg import Bool, String
from geometry_msgs.msg import PoseStamped, PoseArray
from action_msgs.msg import GoalStatus

from nav_limo_interfaces.action import Mission
from nav_limo_interfaces.srv import MissionCommand


# =====================================================
# CLIENT
# =====================================================

class MissionClient(Node):

    def __init__(self):
        super().__init__('mission_client')

        self.cb_group = ReentrantCallbackGroup()

        # GUI SERVER
        self._service = self.create_service(
            MissionCommand,
            '/limo/mission_cmd',
            self._cmd_callback
        )

        # ACTION CLIENT
        self._client = ActionClient(
            self,
            Mission,
            '/limo/mission',
            callback_group=self.cb_group
        )

        # PAUSE TOPIC
        self._pause_pub = self.create_publisher(
            Bool,
            '/limo/mission/pause',
            10
        )

        # VISUALIZATION TOPIC
        self._queued_goals_pub = self.create_publisher(
            PoseArray,
            '/limo/mission/queued_goals',
            10
        )

        # STATE PUBLISHER
        self.state_gui_pub = self.create_publisher(
            String,
            '/limo/mission/state',
            10
        )

        # GOAL PUBLISHER
        self.goal_gui_pub = self.create_publisher(
            String,
            '/limo/mission/goal',
            10
        )

        diagnostics_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )
        self.diagnostics_sub = self.create_subscription(
            String,
            '/limo/mission/diagnostics',
            self._diagnostic_cb,
            diagnostics_qos
        )
        self.controller_attached_sub = self.create_subscription(
            Bool,
            '/limo/mission/controller_attached',
            self._controller_attached_cb,
            diagnostics_qos
        )

        # STATE
        self._goals: List[PoseStamped] = []
        self._active_goal = None

        # SUPPORT VARIABLES
        self.last_state = None
        self.last_goal = None

        self.get_logger().info("MissionClient ready: service /limo/mission_cmd available")
        self._log_terminal_help()

        # Direct interactive console when the node is started with `ros2 run`.
        # Launch files commonly provide no TTY, so in that case commands remain
        # available through /limo/mission_cmd and the GUI.
        if sys.stdin.isatty():
            self._console_thread = threading.Thread(
                target=self._console_loop,
                name="mission_console",
                daemon=True
            )
            self._console_thread.start()
        else:
            self.get_logger().info(
                "Interactive stdin unavailable; use /limo/mission_cmd from another terminal"
            )

    def _console_loop(self):
        self.get_logger().info("Interactive console enabled. Type 'help' for commands.")
        while rclpy.ok():
            try:
                command = input("mission> ").strip()
            except EOFError:
                return
            except KeyboardInterrupt:
                return

            if not command:
                continue

            request = MissionCommand.Request()
            request.command = command
            response = MissionCommand.Response()
            result = self._cmd_callback(request, response)
            outcome = "OK" if result.success else "ERROR"
            print(f"[{outcome}] {result.message}", flush=True)

    def _log_terminal_help(self):
        self.get_logger().info(
            "Type commands directly at the 'mission>' prompt, or from another terminal:\n"
            "  ros2 service call /limo/mission_cmd nav_limo_interfaces/srv/MissionCommand "
            "\"{command: 'add 1.0 2.0 90'}\"\n"
            "  ros2 service call /limo/mission_cmd nav_limo_interfaces/srv/MissionCommand "
            "\"{command: 'list'}\"\n"
            "  ros2 service call /limo/mission_cmd nav_limo_interfaces/srv/MissionCommand "
            "\"{command: 'send'}\"\n"
            "  Commands: add X Y [YAW_DEG], list, clear, send, pause, resume, abort, help"
        )

    def _command_response(self, response, success, message):
        response.success = success
        response.message = message
        if success:
            self.get_logger().info(f"[COMMAND OK] {message}")
        else:
            self.get_logger().warning(f"[COMMAND FAILED] {message}")
        return response

    # =====================================================
    # GUI COMMAND RECEPTION
    # =====================================================

    def _cmd_callback(self, request, response):
        cmd = request.command.strip().split()

        if not cmd:
            return self._command_response(response, False, "empty command; use 'help'")

        action = cmd[0].lower()
        self.get_logger().info(f"[COMMAND] received: {request.command.strip()}")

        try:

            # ---------------- ADD ----------------
            if action == "add":
                if len(cmd) < 3:
                    raise ValueError("usage: add x y yaw")

                x = float(cmd[1])
                y = float(cmd[2])
                yaw = float(cmd[3]) if len(cmd) > 3 else 0.0

                message = self.add(x, y, yaw)
                return self._command_response(response, True, message)

            # ---------------- SEND ----------------
            elif action == "send":
                success, message = self.send()
                return self._command_response(response, success, message)

            # ---------------- ABORT ----------------
            elif action == "abort":
                success, message = self.abort()
                return self._command_response(response, success, message)

            # ---------------- PAUSE ----------------
            elif action == "pause":
                self.pause()
                return self._command_response(response, True, "mission paused")

            # ---------------- RESUME ----------------
            elif action == "resume":
                self.resume()
                return self._command_response(response, True, "mission resumed")

            # ---------------- LIST ----------------
            elif action == "list":
                message = self._log_queued_goals()
                return self._command_response(response, True, message)

            # ---------------- CLEAR ----------------
            elif action == "clear":
                count = len(self._goals)
                self._goals.clear()
                self._publish_queued()
                return self._command_response(
                    response, True, f"cleared {count} queued goal(s)"
                )

            # ---------------- HELP ----------------
            elif action == "help":
                self._log_terminal_help()
                return self._command_response(
                    response,
                    True,
                    "commands: add X Y [YAW_DEG], list, clear, send, pause, resume, abort, help"
                )

            else:
                return self._command_response(
                    response, False, f"unknown command: {action}; use 'help'"
                )

        except Exception as e:
            self.get_logger().error(f"Command error: {e}")
            return self._command_response(response, False, str(e))

    # =====================================================
    # VISUALIZATION HELPER
    # =====================================================

    def _publish_queued(self):
        pa = PoseArray()
        pa.header.stamp = self.get_clock().now().to_msg()
        pa.header.frame_id = 'map'
        pa.poses = [ps.pose for ps in self._goals]
        self._queued_goals_pub.publish(pa)

    # =====================================================
    # GOALS
    # =====================================================

    def add(self, x: float, y: float, yaw_deg: float = 0.0):
        ps = PoseStamped()
        ps.header.frame_id = "map"

        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.position.z = 0.0

        yaw = math.radians(yaw_deg)
        ps.pose.orientation.w = math.cos(yaw / 2.0)
        ps.pose.orientation.z = math.sin(yaw / 2.0)

        self._goals.append(ps)
        self._publish_queued()

        return (
            f"goal {len(self._goals)} added: "
            f"x={x:.2f}, y={y:.2f}, yaw={yaw_deg:.1f} deg"
        )

    def _log_queued_goals(self):
        if not self._goals:
            self.get_logger().info("[QUEUE] empty")
            return "queue empty"

        lines = []
        for index, goal in enumerate(self._goals, start=1):
            q = goal.pose.orientation
            yaw = math.degrees(math.atan2(2.0 * q.w * q.z, 1.0 - 2.0 * q.z * q.z))
            lines.append(
                f"  {index}: x={goal.pose.position.x:.2f}, "
                f"y={goal.pose.position.y:.2f}, yaw={yaw:.1f} deg"
            )
        self.get_logger().info("[QUEUE]\n" + "\n".join(lines))
        return f"{len(self._goals)} queued goal(s); details printed by user_server"

    # =====================================================
    # ACTION CONTROL
    # =====================================================

    def send(self):
        """abort + restart mission"""

        if not self._goals:
            return False, "no goals to send"

        if self._active_goal is not None:
            self.abort(log_if_idle=False)

        if not self._client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Mission action server /mission not available")
            return False, "mission action server /limo/mission not available"

        goal_msg = Mission.Goal()
        goal_msg.goals = list(self._goals)

        goal_count = len(self._goals)
        self.get_logger().info(f"[SEND] sending mission with {goal_count} goal(s)")

        future = self._client.send_goal_async(
            goal_msg,
            feedback_callback=self._feedback_cb
        )

        future.add_done_callback(self._goal_response_cb)
        self._goals.clear()
        self._publish_queued()
        return True, f"mission dispatch requested with {goal_count} goal(s)"

    def abort(self, log_if_idle=True):
        """Cancel current mission"""
        if self._active_goal is None:
            if log_if_idle:
                self.get_logger().warning("[ABORT] no active mission")
            return False, "no active mission to abort"

        self.get_logger().info("[ABORT] cancellation requested")
        self._active_goal.cancel_goal_async()
        self._active_goal = None
        return True, "mission cancellation requested"

    # =====================================================
    # CALLBACK ACTION
    # =====================================================

    def _goal_response_cb(self, future):
        handle = future.result()

        if not handle.accepted:
            self.get_logger().error("[MISSION] rejected by coordinator")
            return

        self._active_goal = handle
        self.get_logger().info("[MISSION] accepted by coordinator")

        result_future = handle.get_result_async()
        result_future.add_done_callback(self._result_cb)

    def _result_cb(self, future):
        try:
            res = future.result()
        except Exception as exc:
            self.get_logger().error(f"[MISSION] result error: {exc}")
            self._active_goal = None
            return
        status = res.status
        detail = res.result.message if res.result else "no result message"

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f"[MISSION] completed: {detail}")
        elif status == GoalStatus.STATUS_CANCELED:
            self.get_logger().info(f"[MISSION] canceled: {detail}")
        else:
            self.get_logger().error(
                f"[MISSION] failed with status={status}: {detail}"
            )

        self._active_goal = None

    def _diagnostic_cb(self, msg):
        self.get_logger().info(f"[SYSTEM HEALTH] {msg.data}")

    def _controller_attached_cb(self, msg):
        if msg.data:
            self.get_logger().info(
                "[CONTROLLER LINK] ATTACHED: planned missions will be executed"
            )
        else:
            self.get_logger().warning(
                "[CONTROLLER LINK] DETACHED: missions will stop after A* planning"
            )

    def _feedback_cb(self, msg):
        fb = msg.feedback
        pose = fb.current_goal

        x = round(pose.pose.position.x, 3)
        y = round(pose.pose.position.y, 3)

        q = pose.pose.orientation
        yaw = round(
            math.degrees(
                math.atan2(
                    2.0 * (q.w * q.z),
                    1.0 - 2.0 * (q.z * q.z)
                )
            ),
            1
        )

        current_state = fb.state
        current_goal = (x, y, yaw)

        if (
            current_state == self.last_state
            and current_goal == self.last_goal
        ):
            return

        # ───────── STATE TOPIC ─────────
        state_msg = String()
        state_msg.data = str(fb.state)
        self.state_gui_pub.publish(state_msg)

        # ───────── GOAL TOPIC ─────────
        goal_msg = String()
        goal_msg.data = f"{x:.2f},{y:.2f},{yaw:.1f}"
        self.goal_gui_pub.publish(goal_msg)

        self.last_state = current_state
        self.last_goal = current_goal
        self.get_logger().info(
            f"[MISSION] state={current_state}, current goal: "
            f"x={x:.2f}, y={y:.2f}, yaw={yaw:.1f} deg"
        )

    # =====================================================
    # PAUSE / RESUME
    # =====================================================

    def pause(self):
        self._pause_pub.publish(Bool(data=True))
        self.get_logger().info("[PAUSE] activated")

    def resume(self):
        self._pause_pub.publish(Bool(data=False))
        self.get_logger().info("[RESUME] activated")


# =====================================================
# MAIN
# =====================================================

def main(args=None):
    rclpy.init(args=args)

    node = MissionClient()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
