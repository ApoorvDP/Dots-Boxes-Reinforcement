import torch

class PyTorchNN(torch.nn.Module):
    
    def __init__(self, n_inputs, network, n_outputs, relu=False):
        super(PyTorchNN, self).__init__()
        network_layers = [torch.nn.Linear(n_inputs, network[0])]
        if len(network) > 1:
            network_layers.append(torch.nn.Tanh() if not relu else torch.nn.ReLU())
            for i in range(len(network)-1):
                network_layers.append(torch.nn.Linear(network[i], network[i+1]))
                network_layers.append(torch.nn.Tanh() if not relu else torch.nn.ReLU())
        network_layers.append(torch.nn.Linear(network[-1], n_outputs))
        #print(network_layers)
        self.model = torch.nn.Sequential(*network_layers)
        self.Xmeans = None
        self.Tmeans = None
    
    def forward(self, X):
        out = self.model(X)
        return out
        
    def train_pytorch(self, X, T, learning_rate, n_iterations, use_SGD=False):
        if self.Xmeans is None:
            self.Xmeans = X.mean(dim=0)
        if self.Tmeans is None:
            self.Tmeans = T.mean(dim=0)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate) if not use_SGD else torch.optim.SGD(self.parameters(), lr=learning_rate)
        loss_func = torch.nn.MSELoss()
        errors = []
        for iteration in range(n_iterations):
            # Forward pass
            outputs = self(X)
            loss = loss_func(outputs, T)
            errors.append(torch.sqrt(loss))
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()    
        return self, errors
    
    def use_pytorch(self, X):
        with torch.no_grad():
            Y = self(X).cpu().numpy() if torch.cuda.is_available() else self(X).numpy()
            return Y