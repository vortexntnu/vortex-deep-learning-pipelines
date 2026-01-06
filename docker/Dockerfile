# ----------------------------------------------------------------------------
# Base Image
# ----------------------------------------------------------------------------
ARG BASE_IMAGE=ros:humble
FROM ${BASE_IMAGE}

# ----------------------------------------------------------------------------
# Setup Arguments & Environment
# ----------------------------------------------------------------------------
USER root
SHELL ["/bin/bash", "-c"]
ARG DEBIAN_FRONTEND=noninteractive
ARG ROS_DISTRO=humble

# Define User Args
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USERNAME=devuser

ENV WORKSPACE=/ros2_ws
WORKDIR ${WORKSPACE}

# ----------------------------------------------------------------------------
# Create non-root user
# ----------------------------------------------------------------------------
RUN groupadd --gid ${GROUP_ID} ${USERNAME} || true && \
    useradd --uid ${USER_ID} --gid ${GROUP_ID} -m -s /bin/bash ${USERNAME} || true && \
    apt-get update && apt-get install -y sudo && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------------------------
# Install System Dependencies
# ----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    git \
    python3-vcstool \
    python3-pip \
    ros-${ROS_DISTRO}-cv-bridge \
    ros-${ROS_DISTRO}-vision-msgs \
    ros-${ROS_DISTRO}-pcl-conversions \
    libopencv-dev \
    libpcl-dev \
    python3-colcon-common-extensions \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip3 install cython

# ----------------------------------------------------------------------------
# Copy & Install Requirements
# ----------------------------------------------------------------------------
COPY requirements.txt ${WORKSPACE}/requirements.txt
RUN pip3 install -r ${WORKSPACE}/requirements.txt

# ----------------------------------------------------------------------------
# Copy Workspace Files
# ----------------------------------------------------------------------------
COPY . ${WORKSPACE}
RUN chown -R ${USER_ID}:${GROUP_ID} ${WORKSPACE}

# ----------------------------------------------------------------------------
# Install ROS dependencies
# ----------------------------------------------------------------------------
RUN rosdep update && rosdep install --from-paths . --ignore-src -r -y

# ----------------------------------------------------------------------------
# User Configuration
# ----------------------------------------------------------------------------
RUN echo "if [ -f /ros2_ws/scripts/bashrc_extra.sh ]; then source /ros2_ws/scripts/bashrc_extra.sh; fi" >> /home/${USERNAME}/.bashrc

# Switch to the user
USER ${USERNAME}
WORKDIR /home/${USERNAME}

CMD ["bash"]
