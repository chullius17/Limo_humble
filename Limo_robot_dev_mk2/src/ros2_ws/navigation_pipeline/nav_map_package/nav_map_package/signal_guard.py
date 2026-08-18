"""Run a child process while reserving SIGINT for the shutdown coordinator."""

import signal
import subprocess
import sys


def main():
    if len(sys.argv) < 2:
        raise RuntimeError('signal_guard requires a command to execute')

    # Put the actual ROS process in a separate process group. ros2 launch sends
    # Ctrl+C to this wrapper, which deliberately keeps the child alive while
    # nav_map asks the map saver to finish its disk write.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    child = subprocess.Popen(sys.argv[1:], start_new_session=True)

    def forward_signal(signum, _frame):
        if child.poll() is None:
            child.send_signal(signum)

    signal.signal(signal.SIGTERM, forward_signal)
    return_code = child.wait()
    raise SystemExit(return_code)


if __name__ == '__main__':
    main()
