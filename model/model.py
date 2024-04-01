import torch
import torch.nn as nn
from torch.nn.functional import one_hot

class PersonalVAD(nn.Module):
    """Personal VAD model class. """

    def __init__(self, input_dim, hidden_dim, num_layers, out_dim, dropout_rate=0.1, use_fc=False, linear=False):
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

        super(PersonalVAD, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.use_fc = use_fc
        self.linear = linear

        # define the model layers...
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # use the original PersonalVAD configuration with one additional layer
        if use_fc:
            self.fc1 = nn.Linear(hidden_dim, hidden_dim)
            self.dropout = nn.Dropout(dropout_rate)
            if not self.linear:
                self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

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

        # pass them through an additional layer if specified...
        if self.use_fc:
            out_packed = self.dropout(self.fc1(out_packed))
            if not self.linear:
                out_packed = self.activation(out_packed)

        out_packed = self.fc2(out_packed)
        return out_packed, hidden

class WPL(nn.Module):
    """Weighted pairwise loss implementation for three classes.

    The weight pairs are interpreted as follows:
    [<ns,tss> ; <ntss,ns> ; <tss,ntss>]

    Target labels contain indices, the model output is a tensor of probabilites for each class.
    (ns, ntss, tss) -> {0, 1, 2}

    For better understanding of the loss function, check out either the original Personal VAD
    paper at https://arxiv.org/abs/1908.04284, or, alternatively, my thesis :)
    """

    def __init__(self, weights=torch.tensor([1.0, 0.5, 1.0])):
        """Initialize the WPL class.

        Args:
            weights (torch.tensor, optional): The weight values for each class pair.
        """

        super(WPL, self).__init__()
        self.weights = weights
        assert len(weights) == 3, "The wpl is defined for three classes only."

    def forward(self, output, target):
        """Compute the WPL for a sequence.

        Args:
            output (torch.tensor): A tensor containing the model predictions.
            target (torch.tensor): A 1D tensor containing the indices of the target classes.

        Returns:
            torch.tensor: A tensor containing the WPL value for the processed sequence.
        """

        output = torch.exp(output)
        label_mask = one_hot(target) > 0.5 # boolean mask
        label_mask_r1 = torch.roll(label_mask, 1, 1) # if ntss, then tss
        label_mask_r2 = torch.roll(label_mask, 2, 1) # if ntss, then ns

        # get the probability of the actual label and the other two into one array
        actual = torch.masked_select(output, label_mask)
        plus_one = torch.masked_select(output, label_mask_r1)
        minus_one = torch.masked_select(output, label_mask_r2)

        # arrays of the first pair weight and the second pair weight used in the equation
        w1 = torch.masked_select(self.weights, label_mask) # if ntss, w1 is <ntss, ns>
        w2 = torch.masked_select(self.weights, label_mask_r1) # if ntss, w2 is <tss, ntss>

        # first pair
        first_pair = w1 * torch.log(actual / (actual + minus_one))
        second_pair = w2 * torch.log(actual / (actual + plus_one))

        # get the negative mean value for the two pairs
        wpl = -0.5 * (first_pair + second_pair)

        # sum and average for minibatch
        return torch.mean(wpl)
