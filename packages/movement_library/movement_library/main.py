from rclpy.node import Node
import logging
import rclpy
import math

from std_msgs.msg import Byte
from std_msgs.msg import Float32
from std_msgs.msg import Int32

from geometry_msgs.msg import Twist, PoseStamped

from example_interfaces.msg import Bool
from passive_sound_localization_msgs.msg import LocalizationResult
from movement_library.logger import setup_logger

from nav2_simple_commander.robot_navigator import BasicNavigator, NavigationResult


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

        # Initialize the navigator for SLAM and navigation
        self.navigator = BasicNavigator()

    def battery_callback(self, msg):
        self.logger.info('battery voltage "%d"' % msg.data)

    def calculate_time_xyz(self, distance, velocity):
        return distance / (velocity * 1.03)

    def calculate_time_ang(self, angle, ang_velocity):
        radians = angle * math.pi / 180
        return radians / (ang_velocity * 0.9881)

    def localizer_callback(self, msg):
        self.logger.info("Got a message")
        # angle = 360-msg.angle
        angle = msg.angle
        distance = msg.distance
        if self.executing:
            pass
        else:
            self.localizationSubscription = {
                "distance": distance,
                "angle": angle,
                "executed": False,
            }
            self.time = 0.0
            self.executing = True
        self.logger.info(f"Got {str(angle)} {str(distance)}")

        # Convert localization result to navigation goal
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose.position.x = distance * math.cos(math.radians(angle))
        goal_pose.pose.position.y = distance * math.sin(math.radians(angle))
        goal_pose.pose.orientation.w = 1.0

        # Send goal to navigator
        self.navigator.goToPose(goal_pose)

    def loop(self):
        # something to stop time from incrementing or reset it when we get a new input
        self.time = self.time + self.loop_time_period

        # self.get_logger().info("time: %d"%(self.time))

        enableMsg = Bool()
        enableMsg.data = True
        self.enable_publisher.publish(enableMsg)

        velocityMsg = Twist()
        velocityMsg.linear.z = 0.0
        velocityMsg.angular.x = 0.0
        velocityMsg.angular.y = 0.0
        velocityMsg.linear.y = 0.0
        velocityMsg.angular.z = 0.0
        velocityMsg.linear.x = 0.0

        if self.localizationSubscription["executed"]:
            return

        # Set speeds
        spdx = 0.3  # Linear speed in m/s
        spdang = 0.5  # Angular speed in rad/s

        # Adjust angular speed based on target direction
        if self.localizationSubscription["angle"] < 0:
            spdang = -spdang

        # Calculate movement times
        time_xyz = self.calculate_time_xyz(
            self.localizationSubscription["distance"], spdx
        )
        time_ang = self.calculate_time_ang(
            self.localizationSubscription["angle"], spdang
        )
        buff = 1
        wait = 2

        time_ang = time_ang + wait
        # print(time_xyz)
        # print(time_ang)
        # print(time_xyz+time_ang+buff)

        # Set angular velocity
        if self.time <= time_ang and self.time > wait:
            velocityMsg.angular.z = spdang
        else:
            velocityMsg.angular.z = 0.0

        # Set linear velocity
        if self.time <= time_xyz + time_ang + buff and self.time > time_ang + buff:
            velocityMsg.linear.x = spdx

        # Stop the robot after movement
        if self.time > (time_xyz + time_ang + buff):
            self.localizationSubscription["distance"] = 0
            self.localizationSubscription["angle"] = 0
            velocityMsg.linear.x = 0.0
            velocityMsg.angular.z = 0.0
            self.time = 0  # not sure if needed
            self.executing = False
            self.localizationSubscription["executed"] = True
        # print(velocityMsg.linear.x)
        # print(velocityMsg.angular.z)

        self.cmd_vel_publisher.publish(velocityMsg)


def main(args=None):
    rclpy.init(args=args)

    node = MovementNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()