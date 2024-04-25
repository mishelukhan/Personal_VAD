import torch
import torch.nn as nn

class TOLSTM(nn.Module):
    """Personal VAD model class. """

    def __init__(self, input_dim, hidden_dim, num_layers, out_dim):
        """PersonalVAD class initializer.

        Args:
            input_dim (int): Input feature vector size.
            hidden_dim (int): LSTM hidden state size.
            num_layers (int): Number of LSTM layers in the model.
            out_dim (int): Number of neurons in the output layer.
            use_fc (bool, optional): Specifies, whether the model should use the
                last fully-connected hidden layer. Defaults to False.
            linear (bool, optional): Specifies the activation function used by the last
                hidden layer. If False, the relu is used, if True, no activation is
                used. Defaults to False.
        """

        super(TOLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.out_dim = out_dim

        # define the model layers...
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, hidden=None):
        """Personal VAD model forward pass method.

        Args:
            x (torch.tensor): Input feature batch.

        Returns:
            tuple: tuple containing:
                out_padded (torch.tensor): Tensor of tensors containing the network predictions.
                    The dimensionality of the output prediction depends on the out_dim attribute.
                hidden (tuple of torch.tensor): Tuple containing the last hidden and cell state
                    values for each processed sequence.
        """

        # lstm pass
        out_packed, hidden = self.lstm(x, hidden)

        out_packed = self.fc(out_packed)
        return out_packed, hidden