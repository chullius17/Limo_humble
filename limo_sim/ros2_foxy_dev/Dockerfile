FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# Set locale and install base development tools.
RUN apt-get update && apt-get install -y --no-install-recommends \
    locales \
    curl \
    gnupg2 \
    lsb-release \
    software-properties-common \
    build-essential \
    cmake \
    git \
    wget \
    python3-pip \
    python3-setuptools \
    nano \
    vim \
    less \
    tree \
    && locale-gen en_US en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# Configure the ROS 2 repository for Ubuntu 20.04 (Focal).
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu focal main" \
    > /etc/apt/sources.list.d/ros2.list

# Install ROS 2 Foxy Desktop, Gazebo integration and workspace build tools.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-foxy-desktop \
    ros-foxy-gazebo-ros-pkgs \
    ros-foxy-robot-localization \
    ros-foxy-rqt-robot-steering \
    ros-foxy-nav2-map-server \
    ros-foxy-nav2-lifecycle-manager \
    ros-foxy-xacro \
    python3-rosdep \
    python3-vcstool \
    python3-colcon-common-extensions \
    python3-argcomplete \
    python3-opencv \
    python3-scipy \
    libopencv-dev \
    libusb-1.0-0-dev \
    libudev-dev \
    libgl1-mesa-glx \
    libglu1-mesa \
    mesa-utils \
    && rm -rf /var/lib/apt/lists/*

# ONNX Runtime 1.10 is compatible with Foxy's Python 3.8 and the pipeline model.
RUN python3 -m pip install --no-cache-dir onnxruntime==1.10.0

# Initialize rosdep and automatically source Foxy in interactive shells.
RUN rosdep init || true
RUN rosdep update --rosdistro foxy
RUN echo "source /opt/ros/foxy/setup.bash" >> /root/.bashrc

WORKDIR /workspace

CMD ["bash"]
