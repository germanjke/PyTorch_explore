import torch
import torch.nn as nn
class Perceptron(nn.Module):
    """ Perceptron is 1 linear layer """
    def __init__(self, input_dim):
        """ Args:
        input_dim (int): size of vector of input features"""
        super(Perceptron, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1)
    def forward(self, x_in):
        """Forward pass of perceptron
        Args:
            x_in (torch.Tensor): tensor of input
                x_in.shape shoule be equal to (batch, num_features)
            Return:
                Result tensor. tensor.shape should be equal to (batch,).
                """
        return torch.sigmoid(self.fc1(x_in)).squeeze()


bce_loss = nn.BCELoss()
sigmoid = nn.Sigmoid()
probabilities = sigmoid(torch.randn(4,1, requires_grad=True))
targets = torch.tensor([1, 0, 1, 0], dtype=torch.float32).view(4,1)
loss = bce_loss(probabilities, targets)

print(probabilities)
print(targets)
print(loss)

#optimizer starts here
import torch.optim as optim

input_dim = 2
lr = 0.001

perceptron = Perceptron(input_dim=input_dim)
bce_loss = nn.BCELoss()
optimizer = optim.Adam(params=perceptron.parameters(), lr=lr)