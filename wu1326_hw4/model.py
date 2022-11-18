import torch
import torch.nn as nn

class MyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MyNetwork, self).__init__()
        """
        In the constructor we define three layers:
        1. A linear layer that maps the input into hidden layer.
        2. A ReLU layer
        3. Another linear layer that makes predictions

        The parameters of this constructor are:
            first parameter: input dimension
            second parameter: dimension of the hidden layers
            third parameter: output dimension
        """
        
        self.fc1 = ...
        self.relu = ...
        self.fc2 = ...
    
    def forward(self, x):
        '''
        This function passes the data x into the model, and returns
        the final output.
        '''
        
        out = ...
        return out
    