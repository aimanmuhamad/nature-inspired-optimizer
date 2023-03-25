import torch
import torch.optim as optim
import numpy as np

class AntColonyOptimization():
    def __init__(self, n_ants, n_iter, lr, rho, beta, Q):
        self.n_ants = n_ants
        self.n_iter = n_iter
        self.lr = lr
        self.rho = rho
        self.beta = beta
        self.Q = Q
        self.pheromone = None
        
    def fit(self, model, loss_fn, X_train, y_train):
        n_input_dim = X_train.shape[1]
        n_output = y_train.shape[1]
        self.pheromone = torch.ones(n_input_dim, n_output) / (n_input_dim * n_output)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        
        for i in range(self.n_iter):
            ants = []
            for j in range(self.n_ants):
                ant = []
                for k in range(n_input_dim):
                    p = self.pheromone[k]
                    p = torch.pow(p, self.beta)
                    p = p / torch.sum(p)
                    idx = np.random.choice(range(n_output), p=p.detach().numpy())
                    ant.append(idx)
                ants.append(ant)
                
            for ant in ants:
                model.zero_grad()
                X_batch = X_train.clone()
                for k, idx in enumerate(ant):
                    X_batch[:,k] = idx
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_train)
                loss.backward()
                optimizer.step()
                
            with torch.no_grad():
                delta_pheromone = torch.zeros_like(self.pheromone)
                for ant in ants:
                    X_batch = X_train.clone()
                    for k, idx in enumerate(ant):
                        X_batch[:,k] = idx
                    y_pred = model(X_batch)
                    loss = loss_fn(y_pred, y_train)
                    gradients = torch.autograd.grad(loss, model.parameters())
                    for k, param in enumerate(model.parameters()):
                        delta_pheromone[k, ant[k]] += gradients[k].abs().mean()
                self.pheromone = (1 - self.rho) * self.pheromone + self.rho * delta_pheromone
                
        return model
    
    def get_config(self):
        config = {'n_ants': self.n_ants,
                  'n_iter': self.n_iter,
                  'lr': self.lr,
                  'rho': self.rho,
                  'beta': self.beta,
                  'Q': self.Q}
        return config
