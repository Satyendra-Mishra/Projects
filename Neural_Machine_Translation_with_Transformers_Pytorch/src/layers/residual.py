from torch import nn

# residual connections
class ResidualConnection(nn.Module):
    def __init__(self, dropout_rate: float):
        super(ResidualConnection, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, output):
        return x + self.dropout(output)
