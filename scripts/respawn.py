#!/usr/bin/env python3
import math
import argparse

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point, Quaternion
from stonefish_ros2.srv import Respawn

def yaw_to_quaternion(yaw_rad: float) -> Quaternion:
    # roll = pitch = 0; solo yaw
    cy = math.cos(yaw_rad * 0.5)
    sy = math.sin(yaw_rad * 0.5)
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = sy
    q.w = cy
    return q

class RespawnClient(Node):
    def __init__(self, service_name='/stonefish_ros2/respawn_robot'):
        super().__init__('respawn_client')
        self.cli = self.create_client(Respawn, service_name)
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f'Waiting for service {service_name}...')

    def call(self, name: str, x: float, y: float, z: float, yaw_deg: float):
        req = Respawn.Request()
        req.name = name

        pose = Pose()
        pose.position = Point(x=x, y=y, z=z)
        pose.orientation = yaw_to_quaternion(math.radians(yaw_deg))
        req.origin = pose

        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

def main():
    parser = argparse.ArgumentParser(description='Respawn a robot in Stonefish')
    parser.add_argument('--name', default='camera_rig', help='Exact name of the robot (case-sensitive)')
    parser.add_argument('--x', type=float, default=0.0)
    parser.add_argument('--y', type=float, default=0.0)
    parser.add_argument('--z', type=float, default=-5.0)
    parser.add_argument('--yaw', type=float, default=0.0, help='Yaw in degrees')
    args = parser.parse_args()

    rclpy.init()
    node = RespawnClient()
    resp = node.call(args.name, args.x, args.y, args.z, args.yaw)
    if resp is None:
        node.get_logger().error('No response received from service')
    else:
        node.get_logger().info(f"success={resp.success} message='{resp.message}'")
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
