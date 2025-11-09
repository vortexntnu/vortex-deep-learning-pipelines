#!/usr/bin/env python3
import math
import argparse
from dataclasses import dataclass

import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point, Quaternion
from stonefish_ros2.srv import Respawn


@dataclass
class Vec3:
    x: float
    y: float
    z: float
    def as_np(self):
        return np.array([self.x, self.y, self.z], dtype=float)


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-9:
        return v
    return v / n


def quat_from_rotation_matrix(R: np.ndarray) -> Quaternion:
    qw = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    qx = math.copysign(math.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2.0, R[2, 1] - R[1, 2])
    qy = math.copysign(math.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2.0, R[0, 2] - R[2, 0])
    qz = math.copysign(math.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2.0, R[1, 0] - R[0, 1])
    q = Quaternion()
    q.x, q.y, q.z, q.w = float(qx), float(qy), float(qz), float(qw)
    return q


def look_at_quat(position: np.ndarray, target: np.ndarray, world_up=np.array([0.0, 0.0, 1.0])):
    forward = normalize(target - position)
    if abs(np.dot(forward, world_up)) > 0.999:
        world_up = np.array([0.0, 1.0, 0.0])
    right = normalize(np.cross(world_up, forward))
    up = normalize(np.cross(forward, right))
    R = np.column_stack((forward, right, up))
    return quat_from_rotation_matrix(R)


class OrbitRespawnNode(Node):
    def __init__(
        self,
        service_name: str,
        robot_name: str,
        p1: Vec3,
        p2: Vec3,
        radius: float,
        angular_speed: float,
        rate_hz: float,
        axial_speed: float,
        bounce_axis: bool,
    ):
        super().__init__("orbit_respawn_client")

        self.cli = self.create_client(Respawn, service_name)
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f"Esperando servicio {service_name} ...")

        self.robot_name = robot_name
        self.radius = radius
        # Axis definition
        P1 = p1.as_np()
        P2 = p2.as_np()
        seg = P2 - P1
        L = float(np.linalg.norm(seg))
        if L < 1e-6:
            raise ValueError("Los puntos p1 y p2 son prácticamente iguales; el eje tiene longitud 0.")
        v_axis = seg / L
        self.P1 = P1
        self.P2 = P2
        self.seg = seg
        self.length = L
        self.axis_dir = v_axis

        # Orthonormal base of the perpendicular plane to the axis: u and w
        any_vec = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(any_vec, v_axis)) > 0.9:
            any_vec = np.array([1.0, 0.0, 0.0])
        self.u = normalize(np.cross(v_axis, any_vec))
        self.w = normalize(np.cross(v_axis, self.u))

        # -------- Upper semicircle (180°) --------
        # Project world_up onto the orbit plane (u, w)
        world_up = np.array([0.0, 0.0, 1.0])
        up_proj = world_up - np.dot(world_up, v_axis) * v_axis
        if np.linalg.norm(up_proj) < 1e-6:
            # If the axis is almost vertical, choose 'u' as the "up" reference
            up_proj = self.u
        up_proj = normalize(up_proj)

        # Center angle of the "up" vector in the base (u, w)
        # We want cos/sin to align the vector with up_proj
        self.ang_center = math.atan2(np.dot(up_proj, self.w), np.dot(up_proj, self.u))

        # Span of 180°: φ ∈ [-π/2, +π/2] with bounce
        self.phi = 0.0
        self.dphi = (angular_speed / rate_hz)  # rad/tick sobre el arco
        self.phi_dir = 1.0  # 1 hacia +π/2, -1 hacia -π/2

        # Axial advance (progress t in [0,1] along the axis P1->P2)
        self.t = 0.0
        self.dt = max(0.0, min(1.0, (axial_speed / rate_hz) / L))  # normalized step per tick
        self.bounce_axis = bounce_axis
        self.t_dir = 1.0  # 1 to P2, -1 to P1

        self.timer = self.create_timer(1.0 / rate_hz, self._tick)

    def _advance_axis(self):
        self.t += self.t_dir * self.dt
        if self.bounce_axis:
            if self.t > 1.0:
                self.t = 1.0
                self.t_dir = -1.0
            elif self.t < 0.0:
                self.t = 0.0
                self.t_dir = 1.0
        else:
            if self.t > 1.0:
                self.t -= 1.0
            elif self.t < 0.0:
                self.t += 1.0

    def _advance_arc(self):
        # Advance φ and bounce at the limits [-π/2, +π/2]
        self.phi += self.phi_dir * self.dphi
        half = 0.5 * math.pi
        if self.phi > half:
            self.phi = half
            self.phi_dir = -1.0
        elif self.phi < -half:
            self.phi = -half
            self.phi_dir = 1.0

    def _tick(self):
        if not hasattr(self, "radius"):
            self.get_logger().warn("radius no estaba definido; usando 3.0 m por defecto")
            self.radius = 3.0
        # Center of the orbit on the axis
        center = self.P1 + self.t * self.seg

        # Absolute angle in the plane (u, w), restricted to 180°
        ang = - self.ang_center - self.phi

        # Point on the semicircle
        pos = center + self.radius * (math.cos(ang) * self.u + math.sin(ang) * self.w)

        # Orientation looking at the center (axis)
        q = look_at_quat(pos, center)

        # Pose
        pose = Pose()
        pose.position = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
        pose.orientation = q

        # Request the Respawn service (uses 'origin')
        req = Respawn.Request()
        req.name = self.robot_name
        req.origin = pose
        self.cli.call_async(req)

        # Advances
        self._advance_arc()
        self._advance_axis()


def main():
    parser = argparse.ArgumentParser(description="Helical semicircle with respawn along an axis (180°)")
    parser.add_argument("--service", default="/stonefish_ros2/respawn_robot")
    parser.add_argument("--name", default="Orca", help="Name of the robot to respawn")
    parser.add_argument("--p1", nargs=3, type=float, default=[1.85, 0.02, 5.53], help="x y z of point 1 of the axis")
    parser.add_argument("--p2", nargs=3, type=float, default=[2.20, -14.02, 7.49], help="x y z of point 2 of the axis")
    parser.add_argument("--radius", type=float, default=3.0, help="Radius of the orbit (m)")
    parser.add_argument("--speed", type=float, default=0.3, help="Angular speed along the arc (rad/s)")
    parser.add_argument("--rate", type=float, default=5.0, help="Frequency of service calls (Hz)")
    parser.add_argument("--axial_speed", type=float, default=0.2, help="Speed along the axis (m/s)")
    parser.add_argument("--bounce_axis", action="store_true", help="On the axis: ping-pong instead of restart")

    args = parser.parse_args()

    rclpy.init()
    node = OrbitRespawnNode(
        service_name=args.service,
        robot_name=args.name,
        p1=Vec3(*args.p1),
        p2=Vec3(*args.p2),
        radius=args.radius,
        angular_speed=args.speed,
        rate_hz=args.rate,
        axial_speed=args.axial_speed,
        bounce_axis=args.bounce_axis,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
