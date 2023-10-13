from torch_geometric.nn import Node2Vec
import torch


def node2vec_ebd(data):
    model = Node2Vec(data.edge_index, embedding_dim=64,
                     walk_length=20,  # lenght of rw
                     context_size=10, walks_per_node=20,
                     num_negative_samples=1,
                     p=200, q=1,  # bias parameters
                     sparse=True).to('cuda')
    loader = model.loader(batch_size=128, shuffle=True, num_workers=4)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)
    model.train()
    for epoch in range(1, 201):
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to('cuda'), neg_rw.to('cuda'))
            loss.backward()
            optimizer.step()

    model.eval()
    return model()
