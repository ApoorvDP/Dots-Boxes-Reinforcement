import torch

class PyTorchNN(torch.nn.Module):
    
    def __init__(self, n_inputs, network, n_outputs, relu):
        super(PyTorchNN, self).__init__()
        self.hidden1 = torch.nn.Linear(n_inputs, network[0])
        self.hidden2 = torch.nn.Linear(network[0], network[1])
        self.hidden3 = torch.nn.Linear(network[1], network[2])
        self.nonlinear = torch.nn.Tanh() if not relu else torch.nn.ReLU()
        self.output = torch.nn.Linear(network[-1], n_outputs)
        self.Xmeans = None
        self.Tmeans = None
    
    def forward(self, X):
        out = self.hidden1(X)
        out = self.nonlinear(out)
        out = self.hidden2(out)
        out = self.nonlinear(out)
        out = self.hidden3(out)
        out = self.nonlinear(out)
        out = self.output(out)
        return out
        
    def train_pytorch(self, X, T, learning_rate, n_iterations, use_SGD):
        if self.Xmeans is None:
            self.Xmeans = X.mean(dim=0)
        if self.Tmeans is None:
            self.Tmeans = T.mean(dim=0)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate) if not use_SGD else torch.optim.SGD(torchnn.parameters(), lr=learning_rate)
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