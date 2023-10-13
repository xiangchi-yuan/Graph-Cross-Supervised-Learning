import random
from copy import deepcopy
from torch_geometric.datasets import Planetoid
from torch_geometric import datasets
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import model.sim_model as sim_model
from model.sim_model import sim_label
from torch import autograd
import torch_geometric.transforms as T
from model.sim_model import GCN, GCNLipMoE
from model.node2vec import node2vec_ebd
import numpy as np



def weak_shot_data(data, new_class, noise_rate):
    new_data = deepcopy(data)
    origin_data = deepcopy(data)
    new_category_nodes = []
    origin_category_nodes = []
    for i in range(len(data.y)):
        if data.y[i] in new_class:
            new_category_nodes.append(i)
            origin_data.train_mask[i], origin_data.val_mask[i], origin_data.test_mask[i] = False, False, False
            origin_data.x[i] = torch.zeros(size=[1, len(origin_data.x[0])])
            if noise_rate > random.random() and new_data.train_mask[i]:
                new_data.y[i] = new_class[random.randint(0, len(new_class) - 1)]
        else:
            origin_category_nodes.append(i)
            new_data.train_mask[i], new_data.val_mask[i], new_data.test_mask[i] = False, False, False
    return origin_data, new_data, origin_category_nodes, new_category_nodes


def weak_shot_data_v2(data, new_class, noise_rate):
    new_data = deepcopy(data)
    origin_data = deepcopy(data)
    new_category_nodes = []
    origin_category_nodes = []
    for i in range(len(data.y)):
        if data.y[i] in new_class:
            new_category_nodes.append(i)
            origin_data.train_mask[i], origin_data.val_mask[i], origin_data.test_mask[i] = False, False, False
            origin_data.x[i] = torch.zeros(size=[1, len(origin_data.x[0])])
            if noise_rate > random.random() and new_data.train_mask[i]:
                new_data.y[i] = new_class[random.randint(0, len(new_class) - 1)]
        else:
            new_category_nodes.append(i)
            origin_category_nodes.append(i)
            # new_data.train_mask[i], new_data.val_mask[i], new_data.test_mask[i] = False, False, False
    return origin_data, new_data, origin_category_nodes, new_category_nodes


def weak_shot_data_v3(data, new_class, noise_rate):
    new_data = deepcopy(data)
    origin_data = deepcopy(data)
    new_category_nodes = []
    origin_category_nodes = []
    for i in range(len(data.y)):
        if data.y[i] in new_class:
            new_category_nodes.append(i)
            origin_data.train_mask[i], origin_data.val_mask[i], origin_data.test_mask[i] = False, False, False
            origin_data.x[i] = torch.zeros(size=[1, len(origin_data.x[0])])
            if noise_rate > random.random() and new_data.train_mask[i]:
                category = new_data.y[i]
                while category == new_data.y[i]:
                    category = new_class[random.randint(0, len(new_class) - 1)]
                new_data.y[i] = category
        else:
            new_category_nodes.append(i)
            origin_category_nodes.append(i)
            # new_data.train_mask[i], new_data.val_mask[i], new_data.test_mask[i] = False, False, False
    return origin_data, new_data, origin_category_nodes, new_category_nodes

def train(data, h_feats, out_feats):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_feats = data.x[0].size()[0]
    model = GCN(in_feats, h_feats, out_feats).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()
    for epoch in range(500):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    pred = model(data).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    print(f'Accuracy: {acc:.4f}')
    return acc


# def weight_loss(out, y, weight):
#     nll_loss = -log_probs[range(len(log_probs)), target]

def train_noise_label(data, nodes, out_feats, sim_scores):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_feats = data.x[0].size()[0]
    model = GCN(in_feats, 32, out_feats).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()
    train_nodes = []
    for i in range(len(nodes)):
        # for i in range(len(data.train_mask)):
        if data.train_mask[nodes[i]]:
            # if data.train_mask[i] and i in nodes:
            train_nodes.append(i)
    # HERE PROBLEM
    sim_scores = sim_scores[train_nodes]
    sim_scores = sim_scores.unsqueeze(1)
    for epoch in range(500):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask] * sim_scores, data.y[data.train_mask])
        # loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    pred = model(data).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    print(f'Accuracy: {acc:.4f}')
    return acc


def train_sim(data, nodes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_feats = data.x[0].size()[0]
    model = sim_model.GCNSim(in_feats).to(device)
    y_sim = torch.tensor(sim_label(data, nodes)).cuda()
    print(y_sim.size(dim=0))
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()
    for epoch in range(100):
        print('epoch: ', epoch)
        optimizer.zero_grad()
        out, x_sim = model(data, nodes)
        loss = F.nll_loss(out, y_sim)

        lip_mat = []
        input = x_sim
        # input.requires_grad_(True)
        for i in range(out.shape[1]):
            v = torch.zeros_like(out)
            v[:, i] = 1
            gradients = autograd.grad(outputs=out, inputs=input, grad_outputs=v,
                                      create_graph=True, retain_graph=True, only_inputs=True)[0]
            grad_norm = torch.norm(gradients, dim=1).unsqueeze(dim=1)
            lip_mat.append(grad_norm)

        # input.requires_grad_(False)
        l_ratio = model.get_l_ratio()
        lip_concat = torch.cat(lip_mat, dim=1)
        lip_con_norm = torch.norm(lip_concat, dim=1)
        lip_loss = torch.max(lip_con_norm)
        loss = loss + l_ratio * lip_loss


        # lip_concat = torch.cat(lip_mat, dim=1)
        # lip_con_norm = torch.norm(lip_concat, dim=1)
        # lip_loss = torch.max(lip_con_norm)
        # loss = loss +  lip_loss

        loss.backward()
        optimizer.step()

    pred, sim_pairs = model(data, nodes)
    pred = pred.argmax(dim=1)
    correct = (pred == y_sim).sum()
    acc = int(correct) / y_sim.size(dim=0)
    print(f'Accuracy: {acc:.4f}')
    return model


def test_sim(model, data, nodes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    y_sim = torch.tensor(sim_label(data, nodes)).cuda()
    print(y_sim.size(dim=0))
    data = data.to(device)
    model.train()
    pred, pairs = model(data, nodes)
    pred = pred.argmax(dim=1)
    correct = (pred == y_sim).sum()
    acc = int(correct) / y_sim.size(dim=0)
    print(f'Accuracy: {acc:.4f}')


def sim_inf(model, data, nodes):
    print('sim inf start')
    model.eval()
    sim_mat, pairs = model(data, nodes)
    sim_mat = F.normalize(sim_mat, p=2, dim=1)
    pair_index = 0
    sim_weight = []
    for i in range(len(nodes)):
        # print(f'Inference process: {i}/{len(nodes)}')
        sim_score = 0
        category_num = 0
        for j in range(50):
            if data.y[nodes[i]] == data.y[nodes[j]]:
                sim_score = sim_score + sim_mat[pair_index][1]
                pair_index += 1
                category_num += 1
        if category_num == 0:
            sim_weight.append(0)
        else:
            sim_weight.append(sim_score / category_num)
        # sim_weight.append(sim_score / category_num)
    sim_scores = torch.tensor(sim_weight).cuda()
    sim_scores = torch.add(sim_scores, 1)
    # print(sim_scores)
    return sim_scores


def main():
    dataset = None
    new_class = None
    dataset_name = 'Wisconsin'
    if dataset_name == 'CiteSeer':
        dataset = Planetoid(root='/tmp/CiteSeer', name='CiteSeer')
        new_class = [0, 1, 2]
    elif dataset_name == 'Cora':
        dataset = Planetoid(root='/tmp/Cora', name='Cora')
        new_class = [0, 1, 2]
    elif dataset_name == 'Actor':
        dataset = datasets.Actor(root='/tmp/Actor')
        new_class = [0, 1]
    elif dataset_name == 'Texas':
        dataset = datasets.WebKB(root='/tmp/Texas', name='Texas')
    elif dataset_name == 'Wisconsin':
        dataset = datasets.WebKB(root='/tmp/Wisconsin', name='Wisconsin') # 32
        new_class = [0, 1]
    elif dataset_name == 'Photo':
        dataset = datasets.Amazon(root='/tmp/Photo', name='Photo')
        new_class = [0, 1, 2, 3]
    elif dataset_name == 'Computers':
        dataset = datasets.Amazon(root='/tmp/Computers', name='Computers')  # 64
        new_class = [0, 1, 2, 3, 4]




    num_classes = dataset.num_classes
    split = T.RandomNodeSplit(num_val=0.1, num_test=0.2)
    dataset = split(dataset[0])


    # pos_emd = node2vec_ebd(dataset).clone().to('cpu').detach()
    # dataset.x = torch.cat((dataset.x, pos_emd), -1)
    origin_data, new_data, origin_category_nodes, new_category_nodes = weak_shot_data_v3(dataset, new_class, 0.3)

    model = train_sim(origin_data, origin_category_nodes)
    test_sim(model, new_data, new_category_nodes)
    sim_scores = sim_inf(model, new_data, new_category_nodes)

    # print('full clean data')
    # train(dataset, 16, num_classes)
    # print('full noise data')
    # train(new_data, 16, num_classes)
    # print('full noise LipMoE')
    # train_noise_label(new_data, new_category_nodes, num_classes, sim_scores)

    num_runs = 5
    accuracies = []
    for i in range(num_runs):
        accuracy = train(dataset, 16, num_classes)
        accuracies.append(accuracy)
    mean_accuracy = np.mean(accuracies)*100
    std_deviation = np.std(accuracies)*100
    result_str = f"{mean_accuracy:.2f} ± {std_deviation:.2f}"
    print(f"full clean data: {result_str}")

    accuracies = []
    for i in range(num_runs):
        accuracy = train(new_data, 16, num_classes)
        accuracies.append(accuracy)
    mean_accuracy = np.mean(accuracies)*100
    std_deviation = np.std(accuracies)*100
    result_str = f"{mean_accuracy:.2f} ± {std_deviation:.2f}"
    print(f"full noise data: {result_str}")

    accuracies = []
    for i in range(num_runs):
        accuracy = train_noise_label(new_data, new_category_nodes, num_classes, sim_scores)
        accuracies.append(accuracy)
    mean_accuracy = np.mean(accuracies)*100
    std_deviation = np.std(accuracies)*100
    result_str = f"{mean_accuracy:.2f} ± {std_deviation:.2f}"
    print(f"full noise data: {result_str}")


if __name__ == "__main__":
    main()
