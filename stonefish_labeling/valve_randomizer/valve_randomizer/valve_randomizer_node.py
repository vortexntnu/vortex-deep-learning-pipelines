import random

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class ValveRandomizer(Node):
    def __init__(self) -> None:
        super().__init__("valve_randomizer")

        self.declare_parameter("valves", ["valve1", "valve2"])
        self.declare_parameter("period", 5.0)
        self.declare_parameter("min_angle", -1.57)
        self.declare_parameter("max_angle", 0.0)

        self.valves = (
            self.get_parameter("valves").get_parameter_value().string_array_value
        )
        period = self.get_parameter("period").value
        self.min_angle = self.get_parameter("min_angle").value
        self.max_angle = self.get_parameter("max_angle").value

        self.publishers_ = {
            valve: self.create_publisher(JointState, f"/{valve}/servos", 10)
            for valve in self.valves
        }

        self.timer = self.create_timer(period, self.tick)
        self.get_logger().info(
            f"Randomizing {list(self.valves)} every {period}s "
            f"in [{self.min_angle}, {self.max_angle}] rad"
        )

    def tick(self) -> None:
        for valve in self.valves:
            angle = random.uniform(self.min_angle, self.max_angle)
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = [f"{valve}/{valve}_joint"]
            msg.position = [angle]
            self.publishers_[valve].publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ValveRandomizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
