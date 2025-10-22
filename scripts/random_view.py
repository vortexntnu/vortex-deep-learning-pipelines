#!/usr/bin/env python3
"""
Random respawn surround an object defined in a .scn (Stonefish).

- If the .scn has world_transform xyz="x y z", use those coordinates.
- If the .scn has world_transform xyz="$(param NAME)", query the parameter NAME
  in /stonefish_ros2/stonefish_simulator via rcl_interfaces/srv/GetParameters.
- Generate an uniform point within a sphere of radius R centered at that point.
- Orientate the yaw towards the target.
- Call the service stonefish_ros2/srv/Respawn.
"""

import re
import math
import random
import argparse
from pathlib import Path
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point, Quaternion
from stonefish_ros2.srv import Respawn
from rcl_interfaces.srv import GetParameters
from rcl_interfaces.msg import ParameterType


# -------------------- Geometric Utilities --------------------

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


def random_point_in_sphere(radius: float) -> Tuple[float, float, float]:
    """Generate an uniform point in the VOLUME of a sphere (not surface)."""
    u = random.random()
    r = radius * (u ** (1.0 / 3.0))
    theta = math.acos(1.0 - 2.0 * random.random())
    phi = 2.0 * math.pi * random.random()
    dx = r * math.sin(theta) * math.cos(phi)
    dy = r * math.sin(theta) * math.sin(phi)
    dz = r * math.cos(theta)
    return dx, dy, dz


# -------------------- .scn Parser --------------------

ROBOT_BLOCK_RE = re.compile(
    r'<robot\b[^>]*name\s*=\s*"(?P<name>[^"]+)"[^>]*>(?P<body>.*?)</robot>',
    re.DOTALL | re.IGNORECASE,
)

WORLD_TRANSFORM_RE = re.compile(
    r'<world_transform\b[^>]*\bxyz\s*=\s*"(?P<xyz>[^"]+)"[^>]*/?>',
    re.DOTALL | re.IGNORECASE,
)

PARAM_EXPR_RE = re.compile(
    r'\$\(\s*param\s+(?P<param>[a-zA-Z0-9_./-]+)\s*\)\s*'
)


def _pick_candidate_name(names):
    """Heuristic to pick a robot if no --target-name is given."""
    priority = ['pipeline', 'dock', 'pool', 'structure', 'seafloor']
    lower = [n.lower() for n in names]
    for p in priority:
        for i, n in enumerate(lower):
            if p in n:
                return names[i]
    return names[0] if names else None


def parse_scn_for_target_center(scn_path: Path,
                                target_name: Optional[str]) -> Tuple[Optional[str], Optional[Tuple[float, float, float]], Optional[str]]:
    """
    Returns a tuple (found_name, center_xyz, param_name)

    - found_name: name of the robot/object found in the .scn (or None if none found).
    - center_xyz: (x, y, z) if the .scn provides numbers in world_transform; None if it uses $(param ...).
    - param_name: name of the parameter if the .scn uses $(param ...); None if it provides numbers.

    Search inside <robot name="...">...</robot> for the first <world_transform xyz="..."> tag and extract xyz.
    """
    text = scn_path.read_text(encoding='utf-8', errors='ignore')

    robots = list(ROBOT_BLOCK_RE.finditer(text))
    if not robots:
        return None, None, None

    chosen = None
    if target_name:
        for m in robots:
            if m.group('name') == target_name:
                chosen = m
                break
        if chosen is None:
            for m in robots:
                if target_name.lower() in m.group('name').lower():
                    chosen = m
                    break
    else:
        names = [m.group('name') for m in robots]
        pick = _pick_candidate_name(names)
        for m in robots:
            if m.group('name') == pick:
                chosen = m
                break

    if chosen is None:
        chosen = robots[0]

    found_name = chosen.group('name')
    body = chosen.group('body')

    wt = WORLD_TRANSFORM_RE.search(body)
    if not wt:
        wt = WORLD_TRANSFORM_RE.search(text)

    if not wt:
        return found_name, None, None

    xyz_raw = wt.group('xyz').strip()

    pm = PARAM_EXPR_RE.search(xyz_raw)
    if pm:
        return found_name, None, pm.group('param')

    parts = xyz_raw.split()
    if len(parts) >= 3:
        try:
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            return found_name, (x, y, z), None
        except ValueError:
            pass

    return found_name, None, None


# -------------------- Client node Respawn --------------------

class RespawnRandomFromScn(Node):
    def __init__(self,
                 service_name: str,
                 param_server_node: str,
                 robot_name: str,
                 scn_path: Path,
                 target_name: Optional[str],
                 radius: float,
                 look_at: bool,
                 min_z: Optional[float],
                 floor_from_param: Optional[str],
                 floor_clearance: float,
                 max_z: Optional[float], 
                 cap_around_z: Optional[float],
                 cap_percent: float,
                 cap_around_center: bool):
        super().__init__('respawn_random_from_scn')

        self.robot_name = robot_name
        self.radius = radius
        self.look_at = look_at
        self.min_z = min_z
        self.floor_from_param = floor_from_param
        self.floor_clearance = floor_clearance
        self.max_z = max_z
        self.cap_around_z = cap_around_z
        self.cap_percent = cap_percent
        self.cap_around_center = cap_around_center

        # Clients
        self.cli_respawn = self.create_client(Respawn, service_name)
        self.cli_get_params = self.create_client(GetParameters, f'{param_server_node}/get_parameters')

        while not self.cli_respawn.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f'Waiting for service {service_name}...')
        while not self.cli_get_params.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f'Waiting for parameter server {param_server_node}...')

        # Parse .scn
        found_name, center_xyz, param_name = parse_scn_for_target_center(scn_path, target_name)
        if found_name:
            self.get_logger().info(f'Objective found in .scn: "{found_name}"')
        else:
            self.get_logger().warn('No <robot name="..."> found in .scn.')

        if self.cap_around_center:
            self.cap_around_z = center_xyz[2]
            self.get_logger().info(f'Upper cap Z of the object: {self.cap_around_z:.3f} (+{self.cap_percent*100:.1f}% allowed)')

        # Sampling center
        if center_xyz is None:
            if param_name is None:
                raise RuntimeError('No xyz or $(param NAME) could be extracted from .scn.')
            center_xyz = self._get_xyz_from_param(param_name)
            if center_xyz is None:
                raise RuntimeError(f'Could not read "{param_name}" from simulator.')
            self.get_logger().info(f'Center from parameter "{param_name}": {center_xyz}')
        else:
            self.get_logger().info(f'Center from explicit xyz in .scn: {center_xyz}')

        # Z of the "floor" (optional)
        self.floor_z = None
        if self.floor_from_param:
            p = self.floor_from_param + "_position"
            p_xyz = self._get_xyz_from_param(p)
            if p_xyz is not None:
                self.floor_z = p_xyz[2]
                self.get_logger().info(f'Z of the floor from parameter "{p}": {self.floor_z:.3f}')

       # Random respawn looking at the target with Z constraints
        self._do_respawn(center_xyz)

    # ---- helpers ----

    def _get_xyz_from_param(self, param_name: str) -> Optional[Tuple[float, float, float]]:
        req = GetParameters.Request()
        req.names = [param_name]
        fut = self.cli_get_params.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        res = fut.result()
        if not res or not res.values:
            return None
        val = res.values[0]

        # DOUBLE_ARRAY [x,y,z]
        if val.type == ParameterType.PARAMETER_DOUBLE_ARRAY and len(val.double_array_value) >= 3:
            arr = val.double_array_value
            return (arr[0], arr[1], arr[2])

        # STRING "[x, y, z]"
        if val.type == ParameterType.PARAMETER_STRING and val.string_value:
            s = val.string_value.strip().lstrip('[').rstrip(']')
            try:
                xs = [float(p) for p in re.split(r'[,\s]+', s) if p]
                if len(xs) >= 3:
                    return (xs[0], xs[1], xs[2])
            except Exception:
                pass
        return None

    def _apply_z_constraints(self, z: float) -> float:
        # 1) clamp with min_z if defined
        if self.min_z is not None:
            z = max(z, self.min_z)
        # 2) clamp with floor z + clearance if defined
        if self.floor_z is not None:
            z = max(z, self.floor_z + self.floor_clearance)
        # 3) clamp with max_z if defined
        if self.max_z is not None:
            z = min(z, self.max_z)
        # 4) clamp with cap-around (reference + percentage)
        if self.cap_around_z is not None:
            z_cap = self.cap_around_z * (1.0 + self.cap_percent)
            z = min(z, z_cap)
        return z


    def _do_respawn(self, center: Tuple[float, float, float]) -> None:
        cx, cy, cz = center
        dx, dy, dz = random_point_in_sphere(self.radius)
        x, y, z = cx + dx, cy + dy, cz + dz

        # Apply Z constraints to avoid "burying" the drone
        z = self._apply_z_constraints(z)

        yaw = math.atan2(cy - y, cx - x) if self.look_at else 0.0

        pose = Pose()
        pose.position = Point(x=x, y=y, z=z)
        pose.orientation = yaw_to_quaternion(yaw)

        req = Respawn.Request()
        req.name = self.robot_name
        req.origin = pose

        fut = self.cli_respawn.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        resp = fut.result()
        if resp:
            self.get_logger().info(
                f"Respawn {self.robot_name} @ ({x:.2f}, {y:.2f}, {z:.2f}) -> success={resp.success} msg='{resp.message}'")
        else:
            self.get_logger().error("Failed to call Respawn service.")


# -------------------- main --------------------

def main():
    parser = argparse.ArgumentParser(description='Random respawn looking at an object from a .scn')
    parser.add_argument('--scn', required=True, help='Path to the .scn file (e.g. pipe.scn, dock.scn, etc.)')
    parser.add_argument('--target-name', default='', help='Exact (or partial) name of the <robot name="..."> inside the .scn')
    parser.add_argument('--robot-name', default='Orca', help='Exact name of the robot to respawn (case-sensitive)')
    parser.add_argument('--radius', type=float, default=5.0, help='Radius of the sphere for random sampling')
    parser.add_argument('--look-at', action='store_true', help='Orient the yaw towards the target')

    # Avoid "burying" under the ground:
    parser.add_argument('--min-z', type=float, default=None,
                        help='Minimum absolute Z allowed for respawn (clamp).')
    parser.add_argument('--floor-from-param', type=str, default=None,
                        help='Parameter prefix for the floor, e.g. "seafloor" → uses seafloor_position.z.')
    parser.add_argument('--floor-clearance', type=float, default=0.5,
                        help='Separation (m) above the ground when using --floor-from-param.')

    parser.add_argument('--service-name', default='/stonefish_ros2/respawn_robot',
                        help='Respawn service (default: /stonefish_ros2/respawn_robot)')
    parser.add_argument('--param-server-node', default='/stonefish_ros2/stonefish_simulator',
                        help='Parameter server node (default: /stonefish_ros2/stonefish_simulator)')
    parser.add_argument('--max-z', type=float, default=5.182732,
                        help='Maximum absolute Z allowed (upper clamp).')
    parser.add_argument('--cap-around-z', type=float, default=None,
                        help='Reference Z for upper cap with percentage (see --cap-percent).')
    parser.add_argument('--cap-percent', type=float, default=0.10,
                        help='Extra fraction allowed over cap-around-z (0.10 = 10%%).')
    parser.add_argument('--cap-around-center', action='store_true',
                        help='Use the Z of the object (center) as reference for the upper cap.')

    args = parser.parse_args()

    scn_path = Path(args.scn)
    if not scn_path.exists():
        raise FileNotFoundError(f'Does not exist: {scn_path}')

    rclpy.init()
    try:
        RespawnRandomFromScn(
            service_name=args.service_name,
            param_server_node=args.param_server_node,
            robot_name=args.robot_name,
            scn_path=scn_path,
            target_name=(args.target_name if args.target_name else None),
            radius=args.radius,
            look_at=args.look_at,
            min_z=args.min_z,
            floor_from_param=args.floor_from_param,
            floor_clearance=args.floor_clearance,
            max_z=args.max_z,
            cap_around_z=args.cap_around_z,
            cap_percent=args.cap_percent,
            cap_around_center=args.cap_around_center
        )
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
