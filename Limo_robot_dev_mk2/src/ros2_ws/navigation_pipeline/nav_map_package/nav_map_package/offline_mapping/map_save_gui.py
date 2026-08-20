"""Small Qt GUI used to request saving the current combined map."""

import sys
import threading

import rclpy
from PyQt5.QtCore import Qt, QTime, pyqtSignal, pyqtSlot
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
from std_msgs.msg import String
from std_srvs.srv import Trigger


class GuiSignals(QWidget):
    """Thread-safe signals from ROS callbacks to the Qt event loop."""

    save_finished = pyqtSignal(bool, str)
    status_received = pyqtSignal(str)


class MapSaveGuiNode(Node):
    """ROS client controlled by the map-save window."""

    def __init__(self, signals):
        super().__init__('map_save_gui')
        self.signals = signals
        self.declare_parameter(
            'save_service',
            '/limo/nav_map_package/offline/map_saver/save_map',
        )
        self.declare_parameter(
            'status_topic',
            '/limo/nav_map_package/offline/map_saver/status',
        )
        service_name = self.get_parameter('save_service').value
        status_topic = self.get_parameter('status_topic').value
        self.save_client = self.create_client(Trigger, service_name)
        status_qos = QoSProfile(
            depth=10,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.status_subscription = self.create_subscription(
            String,
            status_topic,
            self._status_callback,
            status_qos,
        )
        self.get_logger().info(f'Map save GUI targeting {service_name}')

    def _status_callback(self, message):
        self.signals.status_received.emit(message.data)

    def request_save(self):
        """Start a non-blocking map-save request."""
        if not self.save_client.service_is_ready():
            message = f"Service '{self.save_client.srv_name}' is not available"
            self.get_logger().error(message)
            self.signals.save_finished.emit(False, message)
            return False

        future = self.save_client.call_async(Trigger.Request())
        future.add_done_callback(self._save_done)
        return True

    def _save_done(self, future):
        try:
            response = future.result()
            success = bool(response.success)
            message = response.message
        except Exception as exc:
            success = False
            message = f'Map save request failed: {exc}'

        log = self.get_logger().info if success else self.get_logger().error
        log(message)
        self.signals.save_finished.emit(success, message)


class MapSaveWindow(QWidget):
    """Window containing the save button and request status."""

    def __init__(self, node, signals):
        super().__init__()
        self.node = node
        self.setWindowTitle('LIMO Map Saver')
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        self.setMinimumWidth(360)

        layout = QVBoxLayout(self)
        self.status_label = QLabel('Press the button to save the current map.')
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.save_button = QPushButton('SAVE MAP')
        self.save_button.setStyleSheet(
            'background-color: #0056b3; color: white; font-weight: bold; '
            'font-size: 16px; padding: 12px; border-radius: 5px;'
        )
        self.save_button.clicked.connect(self._request_save)
        layout.addWidget(self.save_button)

        terminal_label = QLabel('MAP SAVER TERMINAL')
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

        signals.save_finished.connect(self._show_result)
        signals.status_received.connect(self._append_status)

    @pyqtSlot()
    def _request_save(self):
        self.save_button.setEnabled(False)
        self.status_label.setStyleSheet('')
        self.status_label.setText('Saving map...')
        if not self.node.request_save():
            self.save_button.setEnabled(True)

    @pyqtSlot(bool, str)
    def _show_result(self, success, message):
        color = '#2e7d32' if success else '#c62828'
        self.status_label.setStyleSheet(f'color: {color}; font-weight: bold;')
        self.status_label.setText(message)
        self.save_button.setEnabled(True)

    @pyqtSlot(str)
    def _append_status(self, message):
        timestamp = QTime.currentTime().toString('HH:mm:ss')
        self.terminal.append(f'{timestamp}  {message}')


def main(args=None):
    rclpy.init(args=args)
    app = QApplication(sys.argv)
    signals = GuiSignals()
    node = MapSaveGuiNode(signals)
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    window = MapSaveWindow(node, signals)
    window.show()
    window.raise_()
    window.activateWindow()
    try:
        app.exec_()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        ros_thread.join(timeout=1.0)


if __name__ == '__main__':
    main()
