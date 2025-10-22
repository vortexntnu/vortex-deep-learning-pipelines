# ----------------------------------------------------------------------------
# Base Image
# ----------------------------------------------------------------------------
ARG BASE_IMAGE=ros:humble
FROM ${BASE_IMAGE}

# ----------------------------------------------------------------------------
# Install System & ROS Dependencies
# ----------------------------------------------------------------------------
USER root
SHELL ["/bin/bash", "-c"]
ARG DEBIAN_FRONTEND=noninteractive

ENV WORKSPACE=/ros2_ws
WORKDIR ${WORKSPACE}
ARG ROS_DISTRO=humble

RUN apt-get update && apt-get install -y \
    git \
    python3-rosdep \
    python3-vcstool \
    python3-pip \
    ros-${ROS_DISTRO}-ros-core \
    ros-${ROS_DISTRO}-cv-bridge \
    ros-${ROS_DISTRO}-vision-msgs \
    ros-${ROS_DISTRO}-pcl-conversions \
    libopencv-dev \
    libpcl-dev \
    python3-colcon-common-extensions \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------------------------
# Ensure Cython is installed (for building packages like lapx)
# ----------------------------------------------------------------------------
RUN pip3 install cython

# ----------------------------------------------------------------------------
# Copy Workspace Files
# ----------------------------------------------------------------------------
COPY . ${WORKSPACE}

# ----------------------------------------------------------------------------
# Install Python dependencies (if requirements.txt exists)
# ----------------------------------------------------------------------------
RUN if [ -f requirements.txt ]; then pip3 install -r requirements.txt; fi

# ----------------------------------------------------------------------------
# Install ROS dependencies
# ----------------------------------------------------------------------------
RUN rosdep update && rosdep install --from-paths . --ignore-src -r -y

# ----------------------------------------------------------------------------
# Create non-root user
# ----------------------------------------------------------------------------
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USERNAME=devuser
RUN groupadd --gid ${GROUP_ID} ${USERNAME} || true && \
    useradd --uid ${USER_ID} --gid ${GROUP_ID} -m -s /bin/bash ${USERNAME} || true && \
    apt-get update && apt-get install -y sudo && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

USER ${USERNAME}
WORKDIR /home/${USERNAME}

# ----------------------------------------------------------------------------
# Source ROS setup in bashrc
# ----------------------------------------------------------------------------
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc

# ----------------------------------------------------------------------------
# Default command
# ----------------------------------------------------------------------------
CMD ["bash"]
