import torch
from torch.optim.optimizer import Optimizer

class PelicanOptimization(Optimizer):
    def __init__(self, params, lr=0.01, pelican_size=10, a=0.5, b=0.5, c=0.5, d=0.5):
        self.pelican_size = pelican_size
        self.a = a
        self.b = b
        self.c = c
        self.d = d

        defaults = dict(lr=lr)
        super(PelicanOptimization, self).__init__(params, defaults)

    def step(self, closure=None):
        loss_fn = closure or (lambda: 0)

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                # Initialize pelican's memory
                if not hasattr(p, 'memory'):
                    p.memory = torch.zeros((self.pelican_size,) + p.data.size()).to(p.device)
                    p.memory_loss = torch.zeros(self.pelican_size).to(p.device)

                # Update pelican's memory
                for i in range(self.pelican_size):
                    alpha, beta, gamma, delta = torch.rand(4).to(p.device)
                    memory_diff = self.a * (alpha * (p.memory[i] - p.data)) + \
                                   self.b * (beta * (p.memory.mean(0) - p.data)) + \
                                   self.c * (gamma * (d_p / d_p.norm(p=2))) + \
                                   self.d * (delta * torch.randn_like(p.data))
                    memory_candidate = p.data + memory_diff
                    loss_candidate = loss_fn(memory_candidate)
                    if loss_candidate < p.memory_loss[i]:
                        p.memory[i].data.copy_(memory_candidate)
                        p.memory_loss[i] = loss_candidate

                # Update particle's position
                p.data = p.memory[p.memory_loss.argmin()].clone()

    def zero_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

    def get_config(self):
        config = {
            'lr': self.defaults['lr'],
            'pelican_size': self.pelican_size,
            'a': self.a,
            'b': self.b,
            'c': self.c,
            'd': self.d
        }
        return config
