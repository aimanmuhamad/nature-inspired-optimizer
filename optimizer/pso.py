import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

class ParticleSwarmOptimization(Optimizer):
    def __init__(self, params, lr=0.01, inertia=0.5, cognitive_weight=0.5, social_weight=0.5):
        self.inertia = inertia
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight

        defaults = dict(lr=lr)
        super(ParticleSwarmOptimization, self).__init__(params, defaults)

    def loss_fn(self, model, X, y):
        y_pred = model(X)
        return nn.BCELoss()(y_pred, y)

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                # Initialize particle's velocity
                if not hasattr(p, 'velocity'):
                    p.velocity = torch.zeros_like(p.data)

                # Update particle's velocity
                p.velocity.mul_(self.inertia)
                p.velocity.add_(self.cognitive_weight * torch.randn_like(p.data) * d_p)
                p.velocity.add_(self.social_weight * torch.randn_like(p.data) * p.best_position.data)

                # Update particle's position
                p.data.add_(self.lr * p.velocity)

                # Update particle's best position
                mask = (self.loss_fn(p.data) < self.loss_fn(p.best_position.data)).float().view(-1, 1)
                p.best_position.data.mul_(1 - mask)
                p.best_position.data.add_(mask * p.data)

    def zero_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

    def get_config(self):
        config = {'lr': self.defaults['lr'],
                  'inertia': self.inertia,
                  'cognitive_weight': self.cognitive_weight,
                  'social_weight': self.social_weight}
        return config
