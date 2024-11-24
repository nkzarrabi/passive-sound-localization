from rclpy.node import Node
import logging
import rclpy
import math

from std_msgs.msg import Byte
from std_msgs.msg import Float32
from std_msgs.msg import Int32

from geometry_msgs.msg import Twist

from example_interfaces.msg import Bool
from passive_sound_localization_msgs.msg import LocalizationResult
from movement_library.logger import setup_logger


class MovementNode(Node):
    def __init__(self):
        super().__init__("movement_node")
        setup_logger()
        self.logger = logging.getLogger(__name__)
        self.logger.info("Starting movement_node")
        self.cmd_vel_publisher = self.create_publisher(Twist, "rcm/cmd_vel", 1)
        self.enable_publisher = self.create_publisher(Bool, "rcm/enabled", 1)

        self.batterySubscription = self.create_subscription(
            Float32, "rcm/battery", self.battery_callback, 1
        )
        self.batterySubscription  # prevent warning

        # LOCALIZATION RESULT HERE...
        self.create_subscription(
            LocalizationResult, "localization_results", self.localizer_callback, 10
        )
        self.localizationSubscription = {"distance": 0, "angle": 0, "executed": False}
        # self.create_subscription(...)

        self.loop_time_period = 1.0 / 10.0
        self.loop_timer = self.create_timer(self.loop_time_period, self.loop)
        self.time = 0.0

        # Temporary boolean that is set to true after movement is finished...
        # Do we want the robot to self-adjust to new vocal queues as it's moving?
        self.executing = False
        self.logger = logging.getLogger(__name__)

        # PID parameters
        self.Kp_linear = 0.5  # Proportional gain for distance
        self.Ki_linear = 0.1  # Integral gain for distance
        self.Kd_linear = 0.05  # Derivative gain for distance

        self.Kp_angular = 0.8  # Proportional gain for angle
        self.Ki_angular = 0.2  # Integral gain for angle
        self.Kd_angular = 0.1  # Derivative gain for angle

        # PID state variables
        self.error_linear = 0
        self.integral_linear = 0
        self.prev_error_linear = 0

        self.error_angular = 0
        self.integral_angular = 0
        self.prev_error_angular = 0

    def battery_callback(self, msg):
        self.logger.info('battery voltage "%d"' % msg.data)

    def calculate_time_xyz(self, distance, velocity):
        return distance / (velocity * 1.03)

    def calculate_time_ang(self, angle, ang_velocity):
        radians = angle * math.pi / 180
        return radians / (ang_velocity * 0.9881)

    def localizer_callback(self, msg):
        self.logger.info("Got a message")
        angle = msg.angle
        distance = msg.distance
        if not self.executing:
            self.localizationSubscription = {
                "distance": distance,
                "angle": angle,
                "executed": False,
            }
            self.executing = True
        self.logger.info(f"Got {str(angle)} {str(distance)}")

    def pid_control(self, error, integral, prev_error, Kp, Ki, Kd):
        """Calculate the PID control signal."""
        derivative = (error - prev_error) / self.loop_time_period
        integral += error * self.loop_time_period
        output = Kp * error + Ki * integral + Kd * derivative
        return output, integral

    def loop(self):
        if not self.executing:
            return

        # Enable the robot
        enableMsg = Bool()
        enableMsg.data = True
        self.enable_publisher.publish(enableMsg)

        # Retrieve errors
        distance_error = self.localizationSubscription["distance"]
        angle_error = self.localizationSubscription["angle"]

        # Linear PID control for distance
        linear_control, self.integral_linear = self.pid_control(
            distance_error,
            self.integral_linear,
            self.prev_error_linear,
            self.Kp_linear,
            self.Ki_linear,
            self.Kd_linear,
        )
        self.prev_error_linear = distance_error

        # Angular PID control for angle
        angular_control, self.integral_angular = self.pid_control(
            angle_error,
            self.integral_angular,
            self.prev_error_angular,
            self.Kp_angular,
            self.Ki_angular,
            self.Kd_angular,
        )
        self.prev_error_angular = angle_error

        # Create velocity message
        velocityMsg = Twist()
        velocityMsg.linear.x = max(0.0, min(linear_control, 1.0))  # Clamp between 0 and 1
        velocityMsg.angular.z = max(-1.0, min(angular_control, 1.0))  # Clamp between -1 and 1

        # Stop the robot if within thresholds
        if abs(distance_error) < 0.1:  # Within 10 cm
            velocityMsg.linear.x = 0.0
            self.localizationSubscription["distance"] = 0

        if abs(angle_error) < 1.0:  # Within 1 degree
            velocityMsg.angular.z = 0.0
            self.localizationSubscription["angle"] = 0

        if velocityMsg.linear.x == 0.0 and velocityMsg.angular.z == 0.0:
            self.executing = False
            self.localizationSubscription["executed"] = True

        # Publish velocity message
        self.cmd_vel_publisher.publish(velocityMsg)


def main(args=None):
    rclpy.init(args=args)

    node = MovementNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
