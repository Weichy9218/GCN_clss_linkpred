import torch
import torch.nn as nn
from torch.nn import Parameter


class GCNConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(GCNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index):
        edge_index, norm = self.normalize(x.size(0), edge_index)

        x = x @ self.weight
        out = self.propagate(edge_index, x=x, norm=norm)

        if self.bias is not None:
            out = out + self.bias
        return out

    def normalize(self, num_nodes, edge_index):
        degree = torch.zeros(num_nodes, dtype=torch.float, device=edge_index.device)
        degree.index_add_(0, edge_index[0], torch.ones_like(edge_index[1],
                                                            dtype=torch.float, device=edge_index.device))
        degree_inv_sqrt = 1. / torch.sqrt(degree.clamp(min=1))
        norm = degree_inv_sqrt[edge_index[0]] * degree_inv_sqrt[edge_index[1]]
        return edge_index, norm

    def propagate(self, edge_index, x, norm):
        row, col = edge_index
        out = x[col] * norm.view(-1, 1)
        out = torch.zeros_like(x).scatter_add_(0, row.view(-1, 1).expand_as(out), out)
        return out


if __name__ == '__main__':
    # Example usage:
    # in_channels and out_channels depend on your specific use case.
    # You can adjust them accordingly.
    in_channels = 16
    out_channels = 32
    conv = GCNConv(in_channels, out_channels)
    x = torch.rand((100, in_channels))  # 100 nodes, in_channels features
    edge_index = torch.randint(0, 100, (2, 200))  # 200 undirected edges
    out = conv(x, edge_index)
    print(out.shape)
