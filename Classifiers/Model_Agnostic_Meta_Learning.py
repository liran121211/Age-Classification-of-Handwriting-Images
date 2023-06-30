import random
import sys
import learn2learn as l2l
import numpy as np
import torch
from torch import nn, optim
from torchvision.models import *
import torchvision.models as models
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet
from datetime import date

# CMD run configuration
sys.path.append(r'/home/unknown/python_files/')
from Scripts.BuildDataset import TorchDataLoader, rootDir
from Scripts.ModelArchitectures import *
from Scripts.Enums import *


"""
Demonstrates how to:
    * use the MAML wrapper for fast-adaptation,
    * use the benchmark interface to load Omniglot, and
    * sample tasks and split them in adaptation and evaluation sets.
"""


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways, device, tqdm_desc):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    # Separate data into adaptation/evalutation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots * ways) * 2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    # Adapt the model
    for step in tqdm(range(adaptation_steps), position=3, desc=tqdm_desc, leave=False, colour='BLACK', ncols=80):
        train_error = loss(learner(adaptation_data), adaptation_labels)
        learner.adapt(train_error)

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    valid_error = loss(predictions, evaluation_labels)
    valid_accuracy = accuracy(predictions, evaluation_labels)
    return valid_error, valid_accuracy


def main(ways=2,
        shots=2,
        meta_lr=1e-4,
        fast_lr=1e-2,
        meta_batch_size=32,
        adaptation_steps=2,
        num_iterations=1000,
        cuda=True,
        seed=100,):

    best_acc = 0.0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')
    if cuda:
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda:0')
    print(f"Device {device} is configured.", end='\n')

    # Create model
    model = VGG_()
    model.to(device)
    maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=True)
    opt = optim.Adam(maml.parameters(), meta_lr)
    loss = nn.CrossEntropyLoss()

    # transforms to create few-shot tasks from any PyTorch dataset.
    dataset_train = l2l.data.MetaDataset(TorchDataLoader('QUWI').train.dataset)
    dataset_val = l2l.data.MetaDataset(TorchDataLoader('QUWI').val.dataset)
    dataset_test = l2l.data.MetaDataset(TorchDataLoader('QUWI').test.dataset)

    tasksets_train = l2l.data.TaskDataset(dataset_train,
                                          task_transforms=[
                                              l2l.data.transforms.NWays(dataset_train, ways),
                                              l2l.data.transforms.KShots(dataset_train, 2 * shots),
                                              l2l.data.transforms.LoadData(dataset_train),
                                              # l2l.data.transforms.RemapLabels(dataset_train),
                                              # l2l.data.transforms.ConsecutiveLabels(dataset_train),
                                          ],
                                          num_tasks=len(dataset_train))
    print(f"Train dataset {DatasetsNames.TRAIN_FILE.value} is loaded.", end='\n')


    tasksets_val = l2l.data.TaskDataset(dataset_val,
                                        task_transforms=[
                                            l2l.data.transforms.NWays(dataset_val, ways),
                                            l2l.data.transforms.KShots(dataset_val, 2 * shots),
                                            l2l.data.transforms.LoadData(dataset_val),
                                            # l2l.data.transforms.RemapLabels(dataset_val),
                                            # l2l.data.transforms.ConsecutiveLabels(dataset_val),
                                        ],
                                        num_tasks=len(dataset_val))
    print(f"Validation dataset {DatasetsNames.VALIDATION_FILE.value} is loaded.", end='\n')


    tasksets_test = l2l.data.TaskDataset(dataset_test,
                                        task_transforms=[
                                            l2l.data.transforms.NWays(dataset_test, ways),
                                            l2l.data.transforms.KShots(dataset_test, 2 * shots),
                                            l2l.data.transforms.LoadData(dataset_test),
                                            # l2l.data.transforms.RemapLabels(dataset_test),
                                            # l2l.data.transforms.ConsecutiveLabels(dataset_test),
                                        ],
                                        num_tasks=len(dataset_test))
    print(f"Test dataset {DatasetsNames.TEST_FILE.value} is loaded.", end='\n')


    for iteration in tqdm(range(num_iterations), position=0, desc="Iterations", leave=False, colour='RED', ncols=80):
        opt.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        for task in tqdm(range(meta_batch_size), position=1, desc="Tasks - Train/Val", leave=False, colour='GREEN', ncols=80):
            # Compute meta-training loss
            learner = maml.clone()
            batch = tasksets_train.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               shots,
                                                               ways,
                                                               device,
                                                               'Tasks - Train')
            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()

            # Compute meta-validation loss
            learner = maml.clone()
            batch = tasksets_val.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               shots,
                                                               ways,
                                                               device,
                                                               'Tasks - Validation')
            meta_valid_error += evaluation_error.item()
            meta_valid_accuracy += evaluation_accuracy.item()

        # Print some metrics
        tqdm.write('\n')
        tqdm.write(f'Iteration {iteration}',nolock=True)
        tqdm.write(f'Meta Train Error {meta_train_error / meta_batch_size}')
        tqdm.write(f'Meta Train Accuracy {meta_train_accuracy / meta_batch_size}')
        tqdm.write(f'Meta Valid Error {meta_valid_error / meta_batch_size}')
        tqdm.write(f'Meta Valid Accuracy {meta_valid_accuracy / meta_batch_size}')

        if (meta_train_accuracy / meta_batch_size) > best_acc:
            best_acc = float("{:.2f}".format((meta_train_accuracy / meta_batch_size)))
            torch.save({
                'epoch': iteration,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'adapt_optimizer_state_dict': maml.state_dict(),
                'loss': loss, },
                f'{rootDir()}/Models/{DatasetsNames.MODEL_NAME.value}-{best_acc}-{date.today()}.pt')

        # print into file the statistics.
        with open(rootDir() + r'/Stats/train_QUWI_ENGLISH.txt', 'a+') as f:
            f.write(f'{(meta_train_accuracy / meta_batch_size)} | {(meta_train_error / meta_batch_size)} | '
                    f'{(meta_valid_accuracy / meta_batch_size)} | {(meta_valid_error / meta_batch_size)}\n')

        # Average the accumulated gradients and optimize
        for p in maml.parameters():
            p.grad.data.mul_(1.0 / meta_batch_size)
        opt.step()

    meta_test_error = 0.0
    meta_test_accuracy = 0.0
    for task in tqdm(range(meta_batch_size), position=2, desc="Tasks - Test", leave=False, colour='BLUE', ncols=80):
        # Compute meta-testing loss
        learner = maml.clone()
        batch = tasksets_test.sample()
        evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                           learner,
                                                           loss,
                                                           adaptation_steps,
                                                           shots,
                                                           ways,
                                                           device,
                                                           'Tasks - Test')
        meta_test_error += evaluation_error.item()
        meta_test_accuracy += evaluation_accuracy.item()
    tqdm.write(f'Meta Test Error {meta_test_error / meta_batch_size}')
    tqdm.write(f'Meta Test Accuracy {meta_test_accuracy / meta_batch_size}')


if __name__ == '__main__':

    main()
