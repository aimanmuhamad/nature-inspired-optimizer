import torch
from torch.optim.optimizer import Optimizer

class SalpSwarmOptimization(Optimizer):
    def __init__(self, learning_rate=0.01, num_particles=10, num_iterations=500,lb=-1, ub=1, **kwargs):
        super(SalpSwarmOptimization, self).__init__(**kwargs)
        self.learning_rate = learning_rate
        self.num_particles = num_particles
        self.lb = lb
        self.ub = ub
        self.num_iterations = num_iterations

    def get_updates(self, loss, params):
        grads = torch.autograd.grad(loss, params, create_graph=True)
        self.updates = [torch.zeros_like(p) for p in params]

        particles_position = torch.rand((self.num_particles, len(params))) * (self.ub - self.lb) + self.lb
        particles_velocity = torch.zeros_like(particles_position)

        for i in range(self.num_iterations):
            # Calculate fitness for each particle
            fitness = torch.zeros(self.num_particles)
            for j in range(self.num_particles):
                with torch.no_grad():
                    params[j].copy_(particles_position[j])
                fitness[j] = loss.item()

            # Determine the best particle as the leader
            leader_idx = torch.argmin(fitness)

            # Update the position and velocity of each particle
            for j in range(self.num_particles):
                if j != leader_idx:
                    # Calculate position and velocity based on leader
                    a = torch.rand(1).item()
                    b = torch.rand(1).item()
                    distance = torch.abs(particles_position[leader_idx] - particles_position[j])
                    particles_velocity[j] = a * particles_velocity[j] + b * distance
                    particles_position[j] += particles_velocity[j] * self.learning_rate

                    # Clamp position to feasible range
                    particles_position[j] = torch.clamp(particles_position[j], self.lb, self.ub)

            # Update the leader position
            particles_position[leader_idx] += particles_velocity[leader_idx] * self.learning_rate

            # Clamp position to feasible range
            particles_position[leader_idx] = torch.clamp(particles_position[leader_idx], self.lb, self.ub)

        # Update the parameters of the model with the leader position
        with torch.no_grad():
            params[leader_idx].copy_(particles_position[leader_idx])

        return self.updates

    def get_config(self):
        config = {'learning_rate': self.learning_rate,
                  'num_particles': self.num_particles,
                  'lb': self.lb,
                  'ub': self.ub}
        base_config = super(SalpSwarmOptimization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))