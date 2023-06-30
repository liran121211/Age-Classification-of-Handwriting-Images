import os
import sys
import torch
import numpy as np
import learn2learn as l2l
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
from datetime import date

from efficientnet_pytorch import EfficientNet
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels

# CMD run configuration
sys.path.append(r'/home/unknown/python_files/')
from Scripts.BuildDataset import TorchDataLoader, rootDir
from Scripts.Enums import *


def Efficientnet_B0_():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNet.from_pretrained("efficientnet-b0")
    in_features = model._fc.in_features
    model._fc = nn.Linear(in_features, 2)
    model = torch.nn.DataParallel(model)
    model.to(device)
    return model


def pairwise_distances_logits(a, b):
    n = a.shape[0]
    m = b.shape[0]
    logits = -((a.unsqueeze(1).expand(n, m, -1) -
                b.unsqueeze(0).expand(n, m, -1)) ** 2).sum(dim=2)
    return logits


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(model, batch, ways, shot, query_num, metric=None, device=None):
    if metric is None:
        metric = pairwise_distances_logits

    if device is None:
        device = model.device()

    data, labels = batch
    data = data.to(device)
    labels = labels.to(device)
    n_items = shot * ways

    # Sort data samples by labels
    sort = torch.sort(labels)
    data = data.squeeze(0)[sort.indices].squeeze(0)
    labels = labels.squeeze(0)[sort.indices].squeeze(0)

    # Compute support and query embeddings
    embeddings = model(data)
    support_indices = np.zeros(data.size(0), dtype=bool)
    selection = np.arange(ways) * (shot + query_num)

    for offset in range(shot):
        support_indices[selection + offset] = True

    query_indices = torch.from_numpy(~support_indices)
    support_indices = torch.from_numpy(support_indices)
    support = embeddings[support_indices]
    support = support.reshape(ways, shot, -1).mean(dim=1)
    query = embeddings[query_indices]
    labels = labels[query_indices].long()

    logits = pairwise_distances_logits(query, support)
    loss = F.cross_entropy(logits, labels)
    acc = accuracy(logits, labels)
    return loss, acc


def main(model, test_query, test_shots, test_way, train_query, train_way, shot, max_epoch, device):
    best_acc = 0.0
    train_dataset = l2l.data.MetaDataset(TorchDataLoader('HHD').train.dataset)
    valid_dataset = l2l.data.MetaDataset(TorchDataLoader('HHD').val.dataset)
    test_dataset = l2l.data.MetaDataset(TorchDataLoader('HHD').test.dataset)

    train_dataset = l2l.data.MetaDataset(train_dataset)
    train_transforms = [
        NWays(train_dataset, train_way),
        KShots(train_dataset, train_query + shot),
        LoadData(train_dataset),
        RemapLabels(train_dataset),
    ]
    train_tasks = l2l.data.TaskDataset(train_dataset, task_transforms=train_transforms, num_tasks=-1)
    train_loader = DataLoader(train_tasks, pin_memory=True, shuffle=True)


    valid_dataset = l2l.data.MetaDataset(valid_dataset)
    valid_transforms = [
        NWays(valid_dataset, test_way),
        KShots(valid_dataset, test_query + test_shots),
        LoadData(valid_dataset),
        RemapLabels(valid_dataset),
    ]
    valid_tasks = l2l.data.TaskDataset(valid_dataset,
                                       task_transforms=valid_transforms,
                                       num_tasks=200)
    valid_loader = DataLoader(valid_tasks, pin_memory=True, shuffle=True)


    test_dataset = l2l.data.MetaDataset(test_dataset)
    test_transforms = [
        NWays(test_dataset, test_way),
        KShots(test_dataset, test_query + test_shots),
        LoadData(test_dataset),
        RemapLabels(test_dataset),
    ]
    test_tasks = l2l.data.TaskDataset(test_dataset,
                                      task_transforms=test_transforms,
                                      num_tasks=2000)
    test_loader = DataLoader(test_tasks, pin_memory=True, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    for epoch in tqdm(range(1, max_epoch + 1), position=0, desc="Epochs", leave=False, colour='RED', ncols=80):
        model.train()

        loss_ctr = 0
        n_loss = 0
        n_acc = 0

        for _ in tqdm(range(len(train_tasks)), position=1, desc="Train Stage", leave=False, colour='GREEN', ncols=80):
            batch = next(iter(train_loader))
            loss, acc = fast_adapt(model,
                                   batch,
                                   train_way,
                                   shot,
                                   train_query,
                                   metric=pairwise_distances_logits,
                                   device=device)

            loss_ctr += 1
            n_loss += loss.item()
            n_acc += acc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        lr_scheduler.step()

        print('Epoch {} | Train Loss={:.4f} | Accuracy={:.4f}'.format(epoch, n_loss / loss_ctr, n_acc / loss_ctr))

        model.eval()

        loss_ctr = 0
        n_loss = 0
        n_acc = 0
        for _ in tqdm(range(len(valid_tasks)), position=1, desc="Validation Stage", leave=False, colour='GREEN', ncols=80):
            batch = next(iter(valid_loader))
            loss, acc = fast_adapt(model,
                                   batch,
                                   test_way,
                                   test_shots,
                                   test_query,
                                   metric=pairwise_distances_logits,
                                   device=device)

            loss_ctr += 1
            n_loss += loss.item()
            n_acc += acc

        print('Epoch {} | Valid Loss={:.4f} | Accuracy={:.4f}\n'.format(epoch, n_loss / loss_ctr, n_acc / loss_ctr))

        if (n_acc / loss_ctr) > best_acc:
            best_acc = float("{:.4f}".format((n_acc / loss_ctr)))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'adapt_optimizer_state_dict': lr_scheduler.state_dict(),
                'loss': n_loss, },
                f'{rootDir()}/Models/{DatasetsNames.MODEL_NAME.value}-{best_acc}-{date.today()}.pt')

        # print into file the statistics.
        with open(rootDir() + r'/Stats/train.txt', 'a+') as f:
            f.write(f'{(n_acc / loss_ctr)} | {(n_loss / loss_ctr)}\n')

    loss_ctr = 0
    n_acc = 0

    for _ in tqdm(range(len(test_tasks)), position=1, desc="Test Stage", leave=False, colour='GREEN', ncols=80):
        batch = next(iter(test_loader))
        loss, acc = fast_adapt(model,
                               batch,
                               test_way,
                               test_shots,
                               test_query,
                               metric=pairwise_distances_logits,
                               device=device)
        loss_ctr += 1
        n_acc += acc
        print('Batch {}: {:.2f}({:.2f})'.format(i, n_acc / loss_ctr * 100, acc * 100))


if __name__ == '__main__':
    init_params = {
        'model': Efficientnet_B0_(),
        'max-epoch': 250,  # number of classes
        'test-query': 15,
        'test-shots': 15,  # number of samples
        'test-way': 2,  # number of classes
        'train-query': 15,
        'train-way': 2,  # number of classes
        'shot': 5,
        'use-cuda': True,
        'seed': 42,
        'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    }

    device = init_params['device']
    if init_params['use-cuda']:
        torch.cuda.manual_seed(init_params['seed'])

        main(
            model=init_params['model'],
            max_epoch=init_params['max-epoch'],
            test_query=init_params['test-query'],
            test_shots=init_params['test-shots'],
            test_way=init_params['test-way'],
            train_query=init_params['train-query'],
            train_way=init_params['train-way'],
            shot=init_params['shot'],
            device=init_params['device']
        )
