import torch

class NN(torch.nn.Module):
    
    def __init__(self, standardize, n_inputs, network, n_outputs, relu=False):
        super(NN, self).__init__()
        network_layers = [torch.nn.Linear(n_inputs, network[0])]
        if len(network) > 1:
            network_layers.append(torch.nn.Tanh() if not relu else torch.nn.ReLU())
            for i in range(len(network)-1):
                network_layers.append(torch.nn.Linear(network[i], network[i+1]))
                network_layers.append(torch.nn.Tanh() if not relu else torch.nn.ReLU())
        network_layers.append(torch.nn.Linear(network[-1], n_outputs))
        self.model, self.standardize, self.processed, self.training_time = torch.nn.Sequential(*network_layers), standardize, False, None
        self.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')).double()
    
    def tensor(self, np_array): # Return tensor for Torch
        return torch.from_numpy(np_array.astype('double')).cuda() if torch.cuda.is_available() else torch.from_numpy(np_array.astype('double'))
    
    def standard(self, data, mean, sd):
        return (data-mean)/sd
    
    def process(self, X, T):
        X, T = self.tensor(X), self.tensor(T)
        if not self.processed:
            self.processed = True
            self.Xmeans, self.Xstds, self.Tmeans, self.Tstds = X.mean(dim=0), X.std(dim=0), T.mean(dim=0), T.std(dim=0)
        if not self.standardize: # Return standardized inputs if desired, else return as is
            return X, T
        else:
            return self.standard(X, self.Xmeans, self.Xstds), self.standard(T, self.Tmeans, self.Tstds)
    
    def forward(self, X):
        return self.model(X) # Output of forward pass is passing data through the model
    
    def train_pytorch(self, X, T, reps, batch_size, learning_rate=10**-3, use_SGD=False, verbose=False):
        X, T = self.process(X, T)
        optimizer, loss_func = torch.optim.Adam(self.parameters(), lr=learning_rate) if not use_SGD else torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=0.7, nesterov=True), torch.nn.MSELoss()
        errors, examples = [], X.shape[0]
        for i in range(reps):
            batches = examples//batch_size
            for j in range(batches):
                start, end = j*batch_size, (j+1)*batch_size
                X_batch, T_batch = torch.autograd.Variable(X[start:end, ...], requires_grad=False), torch.autograd.Variable(T[start:end, ...], requires_grad=False)
                optimizer.zero_grad()
                outputs = self.forward(X_batch) # Forward pass
                loss = loss_func(outputs, T_batch)
                loss.backward() # Backward and optimize
                optimizer.step()
            errors.append(torch.sqrt(loss.clone().detach())) # Detach Loss to garbage collect it; error at end of iteration
            if verbose:
                print(f'Iteration {i+1}, Error: {round(errors[-1], 4)}')
        return self, errors
    
    def use_pytorch(self, X):
        X = self.tensor(X)
        with torch.no_grad():
            return self(X).cpu().numpy() if torch.cuda.is_available() else self(X).numpy() # Return Y
