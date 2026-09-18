FROM ubuntu:20.04

ARG USERNAME=ros
ARG USER_UID=1000
ARG USER_GID=1000

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    ROS_DISTRO=foxy \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Base development tools and libraries used by the LIMO workspace.
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash-completion \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    gnupg2 \
    less \
    libboost-program-options-dev \
    libgl1-mesa-glx \
    libglu1-mesa \
    libturbojpeg0-dev \
    libudev-dev \
    libusb-1.0-0-dev \
    locales \
    lsb-release \
    mesa-utils \
    nano \
    python3-pip \
    python3-setuptools \
    software-properties-common \
    sudo \
    tree \
    udev \
    vim \
    wget \
    && locale-gen en_US en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# ROS 2 Foxy targets Ubuntu 20.04 (Focal), independently of the 24.04 host.
RUN curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu focal main" \
      > /etc/apt/sources.list.d/ros2.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
      python3-argcomplete \
      python3-colcon-common-extensions \
      python3-matplotlib \
      python3-numpy \
      python3-opencv \
      python3-pyqt5 \
      python3-pytest \
      python3-rosdep \
      python3-scipy \
      python3-vcstool \
      python3-yaml \
      ros-foxy-desktop \
      ros-foxy-gazebo-ros-pkgs \
      ros-foxy-joint-state-publisher-gui \
      ros-foxy-nav2-bringup \
      ros-foxy-robot-localization \
      ros-foxy-rqt-robot-steering \
      ros-foxy-slam-toolbox \
      ros-foxy-teleop-twist-keyboard \
      ros-foxy-twist-mux \
      ros-foxy-xacro \
    && rm -rf /var/lib/apt/lists/*

# PyTurboJPEG is declared by cv_package. ONNX Runtime is retained for the
# existing perception pipeline and pinned to a Python 3.8-compatible release.
RUN python3 -m pip install --no-cache-dir \
      PyTurboJPEG==1.7.7 \
      onnxruntime==1.10.0

RUN rosdep init 2>/dev/null || true \
    && rosdep update --rosdistro foxy

# Match the host UID/GID so files created in the mounted workspace stay owned
# by the developer instead of root.
RUN groupadd --gid "${USER_GID}" "${USERNAME}" \
    && useradd --uid "${USER_UID}" --gid "${USER_GID}" --create-home --shell /bin/bash "${USERNAME}" \
    && usermod -aG dialout,video,sudo "${USERNAME}" \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > "/etc/sudoers.d/${USERNAME}" \
    && chmod 0440 "/etc/sudoers.d/${USERNAME}" \
    && echo "source /opt/ros/foxy/setup.bash" >> "/home/${USERNAME}/.bashrc" \
    && echo 'test ! -f /workspace/install/setup.bash || source /workspace/install/setup.bash' >> "/home/${USERNAME}/.bashrc"

COPY ros-entrypoint.sh /usr/local/bin/ros-entrypoint
RUN chmod 0755 /usr/local/bin/ros-entrypoint

USER ${USERNAME}
WORKDIR /workspace

ENTRYPOINT ["/usr/local/bin/ros-entrypoint"]
CMD ["bash"]
