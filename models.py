# %%
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch.nn import functional as F
import torch.nn as nn


class GATConv3(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
    ):
        super(GATConv3, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.W = nn.Linear(in_channels, out_channels, bias=False)
        self.a_1 = nn.Linear(out_channels, 1, bias=False)
        self.a_2 = nn.Linear(out_channels, 1, bias=False)

        self.lmbd = nn.Parameter(torch.ones(2708, 1))

        self._set_parameters()

    def forward(self, X, A, gft: GFT):
        K = self.W(X)
        k1 = self.a_1(K)
        k2 = self.a_2(K)

        E = k1 + k2.view(1, -1)
        E = gft.igft(self.lmbd * gft.gft(E)).abs()
        E = F.leaky_relu(E, self.negative_slope)  # possibly remove this

        E_masked = E.masked_fill(A == 0, float("-inf"))
        alpha_mat = F.softmax(E_masked, dim=-1)

        alpha_mat = F.dropout(alpha_mat, p=self.dropout, training=self.training)
        H = alpha_mat @ K

        return H

    def _set_parameters(self):
        nn.init.kaiming_uniform_(self.W.weight)
        nn.init.kaiming_uniform_(self.a_1.weight)
        nn.init.kaiming_uniform_(self.a_2.weight)
        nn.init.kaiming_uniform_(self.lmbd)


class GATConv4(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
    ):
        super(GATConv4, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.W_K = nn.Linear(in_channels, out_channels, bias=False)
        self.W_Q = nn.Linear(in_channels, out_channels, bias=False)
        self.W_V = nn.Linear(in_channels, out_channels, bias=False)
        self.lmbd = nn.Parameter(torch.ones(2708, 1))

        self._set_parameters()

    def forward(self, X, A, gft: GFT):
        K = self.W_K(X)
        Q = self.W_Q(X)
        V = self.W_V(X)

        d_k = K.shape[-1]

        E = (Q @ K.T) / d_k
        E = gft.igft(self.lmbd * gft.gft(E)).abs()

        E_masked = E.masked_fill(A == 0, float("-inf"))

        alpha_mat = F.softmax(E_masked, dim=-1)
        alpha_mat = F.dropout(alpha_mat, p=self.dropout, training=self.training)

        # TODO: bu da fena değil deneylerde kullan
        # alpha_mat = gft.igft(self.lmbd * gft.gft(alpha_mat))
        # alpha_mat = alpha_mat.real

        H = alpha_mat @ V

        return H

    def _set_parameters(self):
        nn.init.kaiming_uniform_(self.W_K.weight)
        nn.init.kaiming_uniform_(self.W_Q.weight)
        nn.init.kaiming_uniform_(self.W_V.weight)
        # nn.init.kaiming_uniform_(self.lmbd)


# %%
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

if __name__ == "__main__":
    device = "cpu"
    name_data = "Cora"
    dataset = Planetoid(root="/tmp/" + name_data, name=name_data)
    dataset.transform = T.NormalizeFeatures()

    print(f"Number of Classes in {name_data}:", dataset.num_classes)
    print(f"Number of Node Features in {name_data}:", dataset.num_node_features)
    data = dataset[0].to(device)

    # Training GATv3
    edge_index = data.edge_index
    adj_coo = torch.sparse_coo_tensor(
        edge_index, values=torch.ones(edge_index.shape[1])
    )

    A = adj_coo.to_dense()
    X = data.x

    # %%
    model = GATvX(
        n_feats=dataset.num_features,
        n_class=dataset.num_classes,
        gat_conv=GATConv4,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    out = model(X, A)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()

# %%
