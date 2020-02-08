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

