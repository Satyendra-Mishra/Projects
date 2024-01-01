from torch import nn

# feed forward block
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout_rate: float) -> None:
        super(FeedForwardBlock, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear_1(x)     # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_ff)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)     # (batch_size, seq_len, d_ff) --> (batch_size, seq_len, d_model)
        return x