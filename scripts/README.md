# Orbit Respawn (180° Helical Path for Stonefish ROS2)

This Python script creates a **helical semicircular motion** (180° arc) of a robot or camera rig along a defined 3D axis inside the **Stonefish Simulator** using ROS 2.

It repeatedly calls the `stonefish_ros2/Respawn` service to reposition (`respawn`) the robot in space, simulating an orbital path without relying on continuous physical dynamics.

---

## Overview

The node moves a robot in a **half-circle (180°)** around an axis defined by two 3D points `p1` and `p2`.
While orbiting, it also travels **along** the axis from `p1` to `p2`, creating a **helical semicircular trajectory** that avoids going below the seafloor.

At each step:

1. The current **center** is calculated on the axis (`p1 → p2`).
2. The robot’s **position** is computed on a semicircle around that center.
3. The robot is **oriented** to always look toward the axis.
4. The pose is sent to the Stonefish simulator using `/stonefish_ros2/respawn_robot`.

---

## How It Works

### Axis definition

The axis is defined by two points:

```bash
--p1 x y z      # start point
--p2 x y z      # end point
```

The vector between them defines the **main direction** of motion. The orbit plane is computed as the plane **perpendicular** to this axis.

### Semicircular motion (180°)

Instead of a full 360° orbit, the script restricts motion to a **half arc** (±90° from the upward direction).

This ensures the drone or camera **never moves below the seafloor**. The motion “bounces” back and forth within this 180° range.

### Axial motion

Simultaneously, the robot **advances along the axis** at a configurable speed.
This produces a **helical** pattern, similar to scanning along a pipe or subsea structure.

When reaching the end (`p2`), the motion can either:

* Restart at `p1` (default), or
* Bounce back (if `--bounce_axis` is set).

---

## Parameters

| Argument        | Type       | Default                         | Description                                 |
| --------------- | ---------- | ------------------------------- | ------------------------------------------- |
| `--service`     | `str`      | `/stonefish_ros2/respawn_robot` | ROS2 Respawn service name                   |
| `--name`        | `str`      | `"Orca"`                        | Name of the robot to respawn                |
| `--p1`          | `float[3]` | `[1.85, 0.02, 5.53]`            | Start of the axis                           |
| `--p2`          | `float[3]` | `[2.20, -14.02, 7.49]`          | End of the axis                             |
| `--radius`      | `float`    | `3.0`                           | Radius of the semicircular orbit (meters)   |
| `--speed`       | `float`    | `0.3`                           | Angular speed in radians per second         |
| `--rate`        | `float`    | `5.0`                           | Frequency of Respawn calls (Hz)             |
| `--axial_speed` | `float`    | `0.2`                           | Translation speed along the axis (m/s)      |
| `--bounce_axis` | flag       | `false`                         | If set, moves back and forth along the axis |

---

## Example Usage

```bash
python3 orbit_respawn.py --name camera_rig \
  --p1 1.85 0.02 5.53 \
  --p2 2.20 -14.02 7.49 \
  --radius 3.0 --speed 0.3 --rate 5 --axial_speed 0.2
```

**Ping-pong mode along the axis:**

```bash
python3 orbit_respawn.py --name camera_rig \
  --p1 1.85 0.02 5.53 --p2 2.20 -14.02 7.49 \
  --radius 3.0 --speed 0.3 --rate 5 --axial_speed 0.2 --bounce_axis
```

---

## Internal Logic Summary

1. **Axis Vector:**
   [
   \vec{v}_{axis} = \frac{p_2 - p_1}{|p_2 - p_1|}
   ]

2. **Perpendicular Plane Basis:**
   Two orthogonal unit vectors `u`, `w` form the plane normal to `v_axis`.

3. **Semicircle Arc:**
   The orbit angle `φ` oscillates in the range ([-π/2, +π/2]), corresponding to a 180° motion above the plane.

4. **Helical Motion:**
   As `φ` sweeps back and forth, the center moves along the axis via normalized progress `t ∈ [0,1]`.

5. **Pose Generation:**
   Each tick computes:
   [
   position = center + radius (\cos φ \cdot u + \sin φ \cdot w)
   ]
   Orientation is calculated via `look_at_quat()` so the robot always faces the axis.

6. **Service Call:**
   The computed pose is sent to Stonefish:

   ```python
   req = Respawn.Request()
   req.name = robot_name
   req.origin = pose
   self.cli.call_async(req)
   ```

---

## License

MIT License — Feel free to use, modify, and adapt.
Developed for **Stonefish-based underwater simulation workflows**.
