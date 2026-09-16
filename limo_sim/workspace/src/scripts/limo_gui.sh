#!/usr/bin/env bash
# Run GUI clients in the PC's existing ROS Foxy container.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
container=${LIMO_GUI_CONTAINER:-limo_sim}
profile=real

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    printf '%s\n' \
        'Usage: bash scripts/limo_gui.sh [--profile real|sim] [launch_argument:=value ...]' \
        '       bash scripts/limo_gui.sh --exec COMMAND [ARG ...]' \
        '' \
        'Default: RViz + Save Map on the PC, using the real robot profile.' \
        'Examples:' \
        '  bash scripts/limo_gui.sh fixed_frame:=odom start_gui:=false' \
        '  bash scripts/limo_gui.sh --profile sim' \
        '  bash scripts/limo_gui.sh --exec ros2 run rqt_gui rqt_gui' \
        '  bash scripts/limo_gui.sh --exec ros2 run rqt_image_view rqt_image_view' \
        '' \
        'Environment: LIMO_GUI_CONTAINER (limo_sim), ROS_DOMAIN_ID (0), DISPLAY.'
    exit 0
fi

if [[ ${1:-} == --profile ]]; then
    profile=${2:-}
    if [[ $profile != real && $profile != sim ]]; then
        printf '%s\n' '--profile must be real or sim.' >&2
        exit 2
    fi
    shift 2
fi

: "${DISPLAY:?Run this script from a graphical terminal on the PC.}"
if [[ $(docker inspect --format '{{.State.Running}}' "$container") != true ]]; then
    printf 'Container %s is not running. Start it first.\n' "$container" >&2
    exit 1
fi

if [[ ${1:-} == --exec ]]; then
    shift
    if [[ $# == 0 ]]; then
        printf '%s\n' '--exec requires a command.' >&2
        exit 2
    fi
    command=("$@")
else
    # The running container can mount a different checkout: stage just this
    # main launch and selected profile, without replacing its workspace.
    package_dir="$script_dir/../ros2_ws/offline_map_package"
    staging_dir=$(docker exec "$container" mktemp -d /tmp/limo-mapping-gui.XXXXXX)
    docker cp "$package_dir/launch/map.launch.py" "$container:$staging_dir/map.launch.py"
    docker cp "$package_dir/config/mapping_$profile.yaml" "$container:$staging_dir/profile.yaml"
    command=(ros2 launch "$staging_dir/map.launch.py"
        mode:=desktop "config_file:=$staging_dir/profile.yaml" "$@")
fi

exec_flags=(-i)
if [[ -t 0 && -t 1 ]]; then
    exec_flags+=(-t)
fi
exec docker exec "${exec_flags[@]}" \
    -e DISPLAY="$DISPLAY" \
    -e QT_X11_NO_MITSHM=1 \
    -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}" \
    -e ROS_LOCALHOST_ONLY=0 \
    "$container" bash -c '
        set -e
        source /opt/ros/foxy/setup.bash
        source /workspace/install/setup.bash
        exec "$@"
    ' limo-gui "${command[@]}"
