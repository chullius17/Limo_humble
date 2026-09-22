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

"""Verify priority and timeout behavior of the velocity multiplexer."""

from geometry_msgs.msg import Twist

from limo_controller.cmd_vel_mux import CommandInput, select_source


def command(priority, timeout, received_at):
    """Create an input containing a valid zero-valued Twist command."""
    return CommandInput(priority, timeout, Twist(), received_at)


def test_teleop_preempts_fresh_autonomy():
    inputs = {
        'autonomy': command(10, 0.5, 1.0),
        'teleop': command(100, 0.5, 1.1),
    }
    assert select_source(inputs, 1.2) == 'teleop'


def test_autonomy_resumes_when_teleop_times_out():
    inputs = {
        'autonomy': command(10, 0.5, 1.6),
        'teleop': command(100, 0.5, 1.0),
    }
    assert select_source(inputs, 1.7) == 'autonomy'


def test_no_source_is_selected_after_all_timeouts():
    inputs = {
        'autonomy': command(10, 0.5, 1.0),
        'teleop': command(100, 0.5, 1.0),
    }
    assert select_source(inputs, 1.6) is None
