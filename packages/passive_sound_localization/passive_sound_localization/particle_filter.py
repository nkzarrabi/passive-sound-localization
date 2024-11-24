import numpy as np
import secrets

class ParticleFilter:
    def __init__(self, num_particles, state_dim, motion_model, measurement_model, resampling_method):
        self.num_particles = num_particles
        self.state_dim = state_dim
        self.particles = np.zeros((num_particles, state_dim))
        self.weights = np.ones(num_particles) / num_particles
        self.motion_model = motion_model
        self.measurement_model = measurement_model
        self.resampling_method = resampling_method

    def initialize_particles(self, init_state, init_cov):
        self.particles = np.random.multivariate_normal(init_state, init_cov, self.num_particles)

    def predict(self, control_input):
        for i in range(self.num_particles):
            self.particles[i] = self.motion_model(self.particles[i], control_input)

    def update(self, measurement):
        for i in range(self.num_particles):
            self.weights[i] = self.measurement_model(self.particles[i], measurement)
        self.weights += 1.e-300  # avoid round-off to zero
        self.weights /= sum(self.weights)  # normalize

    def resample(self):
        indices = self.resampling_method(self.weights)
        self.particles = self.particles[indices]
        self.weights = self.weights[indices]
        self.weights /= sum(self.weights)  # normalize

    def estimate(self):
        return np.average(self.particles, weights=self.weights, axis=0)

def motion_model(state, control_input):
    # Simple random walk model
    noise = np.random.normal(0, 0.1, size=state.shape)
    return state + control_input + noise

def measurement_model(state, measurement):
    # Gaussian likelihood
    error = np.linalg.norm(state - measurement)
    return np.exp(-error**2 / 2.0)

def systematic_resampling(weights):
    N = len(weights)
    positions = (np.arange(N) + secrets.SystemRandom().random()) / N
    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes
