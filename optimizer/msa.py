import torch
import math

class MothSearchAlgorithm(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, a=2, betamin=0.2, lambd=0.1, t=0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if a < 1:
            raise ValueError("Invalid a parameter: {}".format(a))
        if betamin < 0.0 or betamin > 1.0:
            raise ValueError("Invalid betamin parameter: {}".format(betamin))
        if lambd < 0.0 or lambd > 1.0:
            raise ValueError("Invalid lambd parameter: {}".format(lambd))
        if t < 0:
            raise ValueError("Invalid t parameter: {}".format(t))
        defaults = dict(lr=lr, a=a, betamin=betamin, lambd=lambd, t=t)
        super(MothSearchAlgorithm, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(MothSearchAlgorithm, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("MSA does not support sparse gradients")

                lr = group['lr']
                a = group['a']
                betamin = group['betamin']
                lambd = group['lambd']
                t = group['t']

                # Generate new positions of moths
                for i in range(len(p)):
                    r = torch.norm(p[i].data)
                    t0 = t / (1 + lambd * t)
                    for j in range(1, t+1):
                        b = a * torch.exp(-j / t0)
                        theta = 2 * math.pi * torch.rand(1)
                        p[i].data = p[i].data + b * r * torch.exp(-lambd * j) * torch.sin(theta)

                # Find the best moth
                fitness = [-(loss.data)]
                for i in range(1, len(p)):
                    fitness.append(-loss.data)
                fitness_sorted = sorted(fitness, reverse=True)
                best_moth = fitness.index(fitness_sorted[0])

                # Update the position of the best moth
                for i in range(len(p)):
                    if i != best_moth:
                        distance = torch.norm(p[i].data - p[best_moth].data)
                        beta = 1 + (betamin - 1) * (math.exp(-lr * distance**2) - math.exp(-lr))
                        theta = 2 * math.pi * torch.rand(1)
                        p[i].data = p[best_moth].data + beta * torch.exp(-lr * distance**2) * torch.sin(theta)

        return loss

    def get_config(self):
        return {
            'lr': self.defaults['lr'],
            'a': self.defaults['a'],
            'betamin': self.defaults['betamin'],
            'lambd': self.defaults['lambd'],
            't': self.defaults['t']
        }
