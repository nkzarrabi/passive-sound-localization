import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from passive_sound_localization_msgs.msg import LocalizationResult
from .particle_filter import ParticleFilter, motion_model, measurement_model, systematic_resampling

class ParticleFilterNode(Node):
    def __init__(self):
        super().__init__('particle_filter_node')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('particle_filter.num_particles', 1000),
                ('particle_filter.state_dim', 2),
                ('particle_filter.init_state', [0.0, 0.0]),
                ('particle_filter.init_cov', [[1.0, 0.0], [0.0, 1.0]]),
            ]
        )

        self.num_particles = self.get_parameter('particle_filter.num_particles').value
        self.state_dim = self.get_parameter('particle_filter.state_dim').value
        self.init_state = self.get_parameter('particle_filter.init_state').value
        self.init_cov = self.get_parameter('particle_filter.init_cov').value

        self.particle_filter = ParticleFilter(
            num_particles=self.num_particles,
            state_dim=self.state_dim,
            motion_model=motion_model,
            measurement_model=measurement_model,
            resampling_method=systematic_resampling
        )
        self.particle_filter.initialize_particles(self.init_state, self.init_cov)

        self.subscription = self.create_subscription(
            LocalizationResult,
            'localization_results',
            self.listener_callback,
            10
        )
        self.publisher = self.create_publisher(LocalizationResult, 'particle_filter_results', 10)

    def listener_callback(self, msg):
        measurement = [msg.angle, msg.distance]
        self.particle_filter.predict(control_input=[0.0, 0.0])
        self.particle_filter.update(measurement)
        self.particle_filter.resample()
        estimated_state = self.particle_filter.estimate()

        result_msg = LocalizationResult()
        result_msg.angle = float(estimated_state[0])
        result_msg.distance = float(estimated_state[1])
        self.publisher.publish(result_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ParticleFilterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
