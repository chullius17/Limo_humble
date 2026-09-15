"""Small Qt GUI used to start, pause and abort Nav2 path control."""

import signal
import sys
import threading

import rclpy
from PyQt5.QtCore import Qt, QTime, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool


class GuiSignals(QWidget):
    """Thread-safe signals from ROS callbacks to the Qt event loop."""

    request_finished = pyqtSignal(str, bool, bool, str)
    status_received = pyqtSignal(str)
    active_received = pyqtSignal(bool)
    paused_received = pyqtSignal(bool)


class ControlGuiNode(Node):
    """ROS service client controlled by the control window."""

    def __init__(self, signals):
        super().__init__('control_gui')
        self.signals = signals
        self.start_abort_client = self.create_client(
            SetBool,
            '/limo/control/set_active',
        )
        self.pause_resume_client = self.create_client(
            SetBool,
            '/limo/control/set_enabled',
        )
        state_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            String,
            '/limo/control/status',
            lambda msg: self.signals.status_received.emit(msg.data),
            state_qos,
        )
        self.create_subscription(
            Bool,
            '/limo/control/active',
            lambda msg: self.signals.active_received.emit(msg.data),
            state_qos,
        )
        self.create_subscription(
            Bool,
            '/limo/control/paused',
            lambda msg: self.signals.paused_received.emit(msg.data),
            state_qos,
        )

    def request_active(self, active):
        """Request start or abort without blocking the Qt event loop."""
        return self._call_service(
            'active',
            self.start_abort_client,
            active,
        )

    def request_enabled(self, enabled):
        """Request resume or pause without blocking the Qt event loop."""
        return self._call_service(
            'enabled',
            self.pause_resume_client,
            enabled,
        )

    def _call_service(self, kind, client, requested_state):
        if not client.service_is_ready():
            message = f"Service '{client.srv_name}' is not available"
            self.get_logger().error(message)
            self.signals.request_finished.emit(
                kind,
                False,
                requested_state,
                message,
            )
            return False

        request = SetBool.Request()
        request.data = requested_state
        future = client.call_async(request)
        future.add_done_callback(
            lambda result: self._request_done(
                result,
                kind,
                requested_state,
            )
        )
        return True

    def _request_done(self, future, kind, requested_state):
        try:
            response = future.result()
            success = bool(response.success)
            message = response.message
        except Exception as exc:
            success = False
            message = f'Control request failed: {exc}'
        log = self.get_logger().info if success else self.get_logger().error
        log(message)
        self.signals.request_finished.emit(
            kind,
            success,
            requested_state,
            message,
        )


class ControlWindow(QWidget):
    """Window containing pause/resume and start/abort controls."""

    def __init__(self, node, signals):
        super().__init__()
        self.node = node
        self.control_active = False
        self.control_paused = False
        self.control_requested = False

        self.setWindowTitle('LIMO Control')
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        self.setMinimumWidth(360)

        layout = QVBoxLayout(self)
        self.state_label = QLabel('Control state: waiting for status')
        self.state_label.setStyleSheet('font-weight: bold;')
        self.state_label.setWordWrap(True)
        layout.addWidget(self.state_label)

        self.pause_button = QPushButton('PAUSE CONTROL')
        self.pause_button.setEnabled(False)
        self.pause_button.clicked.connect(self._toggle_pause)
        layout.addWidget(self.pause_button)

        self.start_button = QPushButton('START CONTROL')
        self.start_button.clicked.connect(self._toggle_active)
        layout.addWidget(self.start_button)

        terminal_label = QLabel('CONTROL TERMINAL')
        terminal_label.setStyleSheet('font-weight: bold;')
        layout.addWidget(terminal_label)

        self.terminal = QTextEdit()
        self.terminal.setReadOnly(True)
        self.terminal.setMinimumHeight(180)
        self.terminal.setStyleSheet(
            'background-color: #101418; color: #d7e0e7; '
            'font-family: monospace; font-size: 12px; padding: 6px;'
        )
        layout.addWidget(self.terminal)

        signals.request_finished.connect(self._request_finished)
        signals.status_received.connect(self._set_status)
        signals.active_received.connect(self._set_active)
        signals.paused_received.connect(self._set_paused)
        self._refresh_buttons()

    @staticmethod
    def _button_style(color):
        return (
            f'background-color: {color}; color: white; font-weight: bold; '
            'font-size: 16px; padding: 12px; border-radius: 5px;'
        )

    @pyqtSlot()
    def _toggle_active(self):
        requested_state = not self.control_requested
        self.start_button.setEnabled(False)
        if not self.node.request_active(requested_state):
            self.start_button.setEnabled(True)

    @pyqtSlot()
    def _toggle_pause(self):
        requested_state = self.control_paused
        self.pause_button.setEnabled(False)
        if not self.node.request_enabled(requested_state):
            self.pause_button.setEnabled(True)

    @pyqtSlot(str, bool, bool, str)
    def _request_finished(self, kind, success, requested_state, message):
        if success and kind == 'active':
            self.control_requested = requested_state
            if not requested_state:
                self.control_active = False
                self.control_paused = False
        elif success and kind == 'enabled':
            self.control_paused = not requested_state
        self._append_status(message)
        self._refresh_buttons()

    @pyqtSlot(str)
    def _set_status(self, status):
        self.state_label.setText(f'Control state: {status}')
        if status.startswith(('READY:', 'IDLE:', 'ERROR:')):
            self.control_requested = False
        elif status.startswith(('STARTING:', 'ACTIVE:', 'PAUSING:',
                                'PAUSED:', 'RESUMING:', 'ABORTING:')):
            self.control_requested = not status.startswith('ABORTING:')
        color = '#2e7d32'
        if status.startswith(('PAUSED:', 'PAUSING:')):
            color = '#ef6c00'
        elif status.startswith(('ERROR:', 'ABORTING:')):
            color = '#c62828'
        self.state_label.setStyleSheet(
            f'color: {color}; font-weight: bold;'
        )
        self._append_status(status)
        self._refresh_buttons()

    @pyqtSlot(bool)
    def _set_active(self, active):
        self.control_active = active
        self.control_requested = active
        self._refresh_buttons()

    @pyqtSlot(bool)
    def _set_paused(self, paused):
        self.control_paused = paused
        if paused:
            self.control_requested = True
        self._refresh_buttons()

    def _refresh_buttons(self):
        if self.control_paused:
            self.pause_button.setText('RESUME CONTROL')
            self.pause_button.setStyleSheet(self._button_style('#2e7d32'))
        else:
            self.pause_button.setText('PAUSE CONTROL')
            self.pause_button.setStyleSheet(self._button_style('#ef6c00'))
        self.pause_button.setEnabled(self.control_requested)

        if self.control_requested:
            self.start_button.setText('ABORT CONTROL')
            self.start_button.setStyleSheet(self._button_style('#c62828'))
        else:
            self.start_button.setText('START CONTROL')
            self.start_button.setStyleSheet(self._button_style('#0056b3'))
        self.start_button.setEnabled(True)

    @pyqtSlot(str)
    def _append_status(self, message):
        timestamp = QTime.currentTime().toString('HH:mm:ss')
        self.terminal.append(f'{timestamp}  {message}')


def main(args=None):
    """Run the Qt control GUI and its ROS client node."""
    rclpy.init(args=args)
    app = QApplication(sys.argv)
    shutdown_requested = threading.Event()

    def request_shutdown(_signum=None, _frame=None):
        if shutdown_requested.is_set():
            return
        shutdown_requested.set()
        app.quit()

    previous_sigint_handler = signal.signal(signal.SIGINT, request_shutdown)
    previous_sigterm_handler = signal.signal(signal.SIGTERM, request_shutdown)
    signal_dispatch_timer = QTimer()
    signal_dispatch_timer.timeout.connect(lambda: None)
    signal_dispatch_timer.start(20)

    signals = GuiSignals()
    node = ControlGuiNode(signals)
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    window = ControlWindow(node, signals)
    window.show()
    window.raise_()
    window.activateWindow()
    try:
        app.exec_()
    finally:
        signal_dispatch_timer.stop()
        signal.signal(signal.SIGINT, previous_sigint_handler)
        signal.signal(signal.SIGTERM, previous_sigterm_handler)
        if rclpy.ok():
            rclpy.shutdown()
        ros_thread.join(timeout=1.0)
        node.destroy_node()


if __name__ == '__main__':
    main()
