import torch
import torch.optim as optim
import numpy as np

class HarrisHawkOptimization(optim.Optimizer):
    def __init__(self, params, lr=0.01, num_harmony=10, lb=-1, ub=1):
        self.lr = lr
        self.num_harmony = num_harmony
        self.lb = lb
        self.ub = ub

        defaults = dict(lr=lr, num_harmony=num_harmony, lb=lb, ub=ub)
        super(HarrisHawkOptimization, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state['iteration'] = 0
                    state['position'] = np.random.uniform(self.lb, self.ub, size=p.data.shape)
                    state['pitch'] = np.random.uniform(self.lb, self.ub, size=p.data.shape)

                position = state['position']
                pitch = state['pitch']

                # Calculate fitness for each harmony
                fitness = np.zeros(self.num_harmony)
                for j in range(self.num_harmony):
                    new_position = np.zeros_like(position)
                    for k in range(len(position)):
                        rand = np.random.rand()
                        if rand < 0.5:
                            new_position[k] = position[k] + np.random.rand() * (pitch[k] - position[k])
                        else:
                            new_position[k] = pitch[k] + np.random.rand() * (position[k] - pitch[k])
                        new_position[k] = np.clip(new_position[k], self.lb, self.ub)

                    with torch.no_grad():
                        p.data = torch.tensor(new_position, device=p.device, dtype=p.dtype)
                        fitness[j] = closure().item()

                    if fitness[j] < state['best_fitness']:
                        state['best_fitness'] = fitness[j]
                        state['best_position'] = new_position

                # Determine the best harmony as the pitch
                pitch = state['best_position']

                # Update the position of each harmony
                new_position = np.zeros_like(position)
                for k in range(len(position)):
                    rand = np.random.rand()
                    if rand < 0.5:
                        new_position[k] = position[k] + np.random.rand() * (pitch[k] - position[k])
                    else:
                        new_position[k] = pitch[k] + np.random.rand() * (position[k] - pitch[k])
                    new_position[k] = np.clip(new_position[k], self.lb, self.ub)

                p.data = torch.tensor(new_position, device=p.device, dtype=p.dtype)

                state['position'] = new_position
                state['pitch'] = pitch
                state['iteration'] += 1

        return loss

    def get_config(self):
        return {
            'lr': self.lr,
            'num_harmony': self.num_harmony,
            'lb': self.lb,
            'ub': self.ub
        }
