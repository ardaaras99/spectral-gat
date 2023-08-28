# %%
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

from models import GATvX, GATConv4, GATv1, GATConv3
from tqdm.auto import tqdm
from gft_torch.gft import GFT

# %%

name_data = "Cora"
dataset = Planetoid(root="/tmp/" + name_data, name=name_data)
dataset.transform = T.NormalizeFeatures()

print(f"Number of Classes in {name_data}:", dataset.num_classes)
print(f"Number of Node Features in {name_data}:", dataset.num_node_features)
data = dataset[0]

# Training GATv3
edge_index = data.edge_index
adj_coo = torch.sparse_coo_tensor(edge_index, values=torch.ones(edge_index.shape[1]))
A = adj_coo.to_dense()
X = data.x

device = "cpu"
A = A.to(device)
X = X.to(device)

gft = GFT(A)

model = GATvX(
    n_feats=dataset.num_features,
    n_class=dataset.num_classes,
    gat_conv=GATConv3,
    gft=gft,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

best_test_acc = 0

t = tqdm(range(1000))

for epoch in t:
    model.train()
    out = model(X, A)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    # print(f"Epoch: {epoch}, Loss: {loss}")
    model.eval()
    _, pred = model(X, A).max(dim=1)
    correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / data.test_mask.sum().item()

    if acc > best_test_acc:
        best_test_acc = acc

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    t.set_description(f"Loss: {loss:.4f}, Best Test Accuracy: {best_test_acc:.3f}")

# %%
model = GATv1(
    num_features=dataset.num_features,
    num_classes=dataset.num_classes,
).to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    if epoch % 20 == 0:
        print(f"Epoch: {epoch}, Loss: {loss}")

    loss.backward()
    optimizer.step()


model.eval()
_, pred = model(data).max(dim=1)
correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct / data.test_mask.sum().item()
print("Test Accuracy: {:.4f}".format(acc))

# %%
