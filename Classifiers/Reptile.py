import os
import sys
import copy
import torch
import random
import numpy as np
import learn2learn as l2l

from torch import nn, optim
from torchvision.models import *
from tqdm import tqdm
from datetime import date
from torch.optim.lr_scheduler import StepLR

# CMD run configuration
sys.path.append(r'/home/unknown/python_files/')
from Scripts.BuildDataset import TorchDataLoader, rootDir
from Scripts.ModelArchitectures import *
from Scripts.Enums import *


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(batch, learner, adapt_opt, loss, adaptation_steps, shots, ways, batch_size, device):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    # Separate data into adaptation/evalutation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots * ways)] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    # Adapt the model
    for step in range(adaptation_steps):
        idx = torch.randint(
            adaptation_data.size(0),
            size=(batch_size,)
        )
        adapt_X = adaptation_data[idx]
        adapt_y = adaptation_labels[idx]
        adapt_opt.zero_grad()
        error = loss(learner(adapt_X), adapt_y)
        error.backward()
        adapt_opt.step()

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    valid_error = loss(predictions, evaluation_labels)
    valid_error /= len(evaluation_data)
    valid_accuracy = accuracy(predictions, evaluation_labels)
    return valid_error, valid_accuracy


def main(
        ways=4,
        train_shots=5,
        test_shots=5,
        meta_lr=1.0,
        meta_lr_final=1.0,
        meta_bsz=5,
        fast_lr=0.001,
        train_bsz=5,
        test_bsz=5,
        train_steps=8,
        test_steps=50,
        iterations=100000,
        test_interval=100,
        cuda=1,
        seed=42,
):
    # Create model
    best_acc = 0.0
    cuda = bool(cuda)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')
    if cuda and torch.cuda.device_count():
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')
    print(f"Device {device} is configured.", end='\n')

    continue_full_train = False
    model = Efficientnet_B0_(4)
    model.to(device)
    opt = torch.optim.SGD(model.module._fc.parameters(), meta_lr)
    adapt_opt = torch.optim.Adam(model.module._fc.parameters(), lr=fast_lr, betas=(0, 0.999))
    adapt_opt_state = adapt_opt.state_dict()
    loss = torch.nn.CrossEntropyLoss(reduction='mean')

    train_dataset = l2l.data.MetaDataset(
        TorchDataLoader('KHATT_COMBINED_PARAGRAPH', 4).train['age'].dataset)  # any PyTorch dataset
    print(f"Train dataset {DatasetsNames.TRAIN_FILE.value} is loaded.", end='\n')

    valid_dataset = l2l.data.MetaDataset(
        TorchDataLoader('KHATT_COMBINED_PARAGRAPH', 4).val['age'].dataset)  # any PyTorch dataset
    print(f"Validation dataset {DatasetsNames.VALIDATION_FILE.value} is loaded.", end='\n')

    test_dataset = l2l.data.MetaDataset(
        TorchDataLoader('KHATT_COMBINED_PARAGRAPH', 4).test['age'].dataset)  # any PyTorch dataset
    print(f"Test dataset {DatasetsNames.TEST_FILE.value} is loaded.", end='\n')

    train_tasks = l2l.data.TaskDataset(train_dataset,
                                       task_transforms=[
                                           l2l.data.transforms.NWays(train_dataset, ways),
                                           l2l.data.transforms.KShots(train_dataset, train_shots),
                                           l2l.data.transforms.LoadData(train_dataset),
                                           l2l.data.transforms.RemapLabels(train_dataset),
                                           l2l.data.transforms.ConsecutiveLabels(train_dataset),
                                       ],
                                       )

    valid_tasks = l2l.data.TaskDataset(valid_dataset,
                                       task_transforms=[
                                           l2l.data.transforms.NWays(valid_dataset, ways),
                                           l2l.data.transforms.KShots(valid_dataset, test_shots),
                                           l2l.data.transforms.LoadData(valid_dataset),
                                           l2l.data.transforms.RemapLabels(valid_dataset),
                                           l2l.data.transforms.ConsecutiveLabels(valid_dataset),
                                       ],
                                       )

    test_tasks = l2l.data.TaskDataset(test_dataset,
                                      task_transforms=[
                                          l2l.data.transforms.NWays(test_dataset, ways),
                                          l2l.data.transforms.KShots(test_dataset, test_shots),
                                          l2l.data.transforms.LoadData(test_dataset),
                                          l2l.data.transforms.RemapLabels(test_dataset),
                                          l2l.data.transforms.ConsecutiveLabels(test_dataset),
                                      ],
                                      )
    train_inner_errors = []
    train_inner_accuracies = []
    valid_inner_errors = []
    valid_inner_accuracies = []
    test_inner_errors = []
    test_inner_accuracies = []

    for iteration in tqdm(range(iterations), position=0, desc="Iterations", leave=False, colour='RED', ncols=80):
        opt.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        meta_test_error = 0.0
        meta_test_accuracy = 0.0

        if iteration > 20 and not continue_full_train:
            continue_full_train = True

            for param in self.model.module.parameters():
                param.requires_grad = True

            opt = torch.optim.SGD(model.module.parameters(), lr=learning_rate)
            adapt_opt = torch.optim.Adam(model.module.parameters(), lr=fast_lr, betas=(0, 0.999))

        # anneal meta-lr
        frac_done = float(iteration) / iterations
        new_lr = frac_done * meta_lr_final + (1 - frac_done) * meta_lr
        for pg in opt.param_groups:
            pg['lr'] = new_lr

        # zero-grad the parameters
        if iteration <= 20:
            for p in model.module._fc.parameters():
                p.grad = torch.zeros_like(p.data)
        else:
            for p in model.module.parameters():
                p.grad = torch.zeros_like(p.data)

        for task in tqdm(range(meta_bsz), position=1, desc="Meta Tasks", leave=False, colour='GREEN', ncols=80):
            # Compute meta-training loss
            learner = copy.deepcopy(model)

            if iteration <= 20:
                adapt_opt = torch.optim.Adam(model.module._fc.parameters(), lr=fast_lr, betas=(0, 0.999))
            else:
                adapt_opt = torch.optim.Adam(model.module.parameters(), lr=fast_lr, betas=(0, 0.999))

            adapt_opt.load_state_dict(adapt_opt_state)
            batch = train_tasks.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               adapt_opt,
                                                               loss,
                                                               train_steps,
                                                               train_shots,
                                                               ways,
                                                               train_bsz,
                                                               device)
            adapt_opt_state = adapt_opt.state_dict()
            if iteration <= 20:
                for p, l in zip(model.module.module._fc.parameters(), learner.parameters()):
                    p.grad.data.add_(-1.0, l.data)
            else:
                for p, l in zip(model.module.parameters(), learner.parameters()):
                    p.grad.data.add_(-1.0, l.data)

            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()

            if iteration % test_interval == 0:
                # Compute meta-validation loss
                learner = copy.deepcopy(model)

                if iteration <= 20:
                    adapt_opt = torch.optim.Adam(model.module._fc.parameters(), lr=fast_lr, betas=(0, 0.999))
                else:
                    adapt_opt = torch.optim.Adam(model.module.parameters(), lr=fast_lr, betas=(0, 0.999))

                adapt_opt.load_state_dict(adapt_opt_state)
                batch = valid_tasks.sample()
                evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                                   learner,
                                                                   adapt_opt,
                                                                   loss,
                                                                   test_steps,
                                                                   test_shots,
                                                                   ways,
                                                                   test_bsz,
                                                                   device)
                meta_valid_error += evaluation_error.item()
                meta_valid_accuracy += evaluation_accuracy.item()

                # Compute meta-testing loss
                learner = copy.deepcopy(model)

                if iteration <= 20:
                    adapt_opt = torch.optim.Adam(model.module._fc.parameters(), lr=fast_lr, betas=(0, 0.999))
                else:
                    adapt_opt = torch.optim.Adam(model.module.parameters(), lr=fast_lr, betas=(0, 0.999))

                adapt_opt.load_state_dict(adapt_opt_state)
                batch = test_tasks.sample()
                evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                                   learner,
                                                                   adapt_opt,
                                                                   loss,
                                                                   test_steps,
                                                                   test_shots,
                                                                   ways,
                                                                   test_bsz,
                                                                   device)
                meta_test_error += evaluation_error.item()
                meta_test_accuracy += evaluation_accuracy.item()

        # Print some metrics
        print('\n')
        print('Iteration', iteration)
        print('Meta Train Error', meta_train_error / meta_bsz)
        print('Meta Train Accuracy', meta_train_accuracy / meta_bsz)
        if iteration % test_interval == 0:
            print('Meta Valid Error', meta_valid_error / meta_bsz)
            print('Meta Valid Accuracy', meta_valid_accuracy / meta_bsz)
            print('Meta Test Error', meta_test_error / meta_bsz)
            print('Meta Test Accuracy', meta_test_accuracy / meta_bsz)

        if (meta_train_accuracy / meta_bsz) > best_acc:
            best_acc = float("{:.2f}".format((meta_train_accuracy / meta_bsz)))
            torch.save({
                'epoch': iteration,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'adapt_optimizer_state_dict': adapt_opt.state_dict(),
                'loss': loss, },
                f'{rootDir()}/Models/{DatasetsNames.MODEL_NAME.value}-{best_acc}-{date.today()}.pt')

        # print into file the statistics.
        with open(rootDir() + r'/Stats/train.txt', 'a+') as f:
            f.write(f'{(meta_train_accuracy / meta_bsz)} | {(meta_train_error / meta_bsz)} | '
                    f'{(meta_valid_accuracy / meta_bsz)} | {(meta_valid_error / meta_bsz)} | '
                    f'{(meta_test_accuracy / meta_bsz)} | {(meta_test_error / meta_bsz)}\n')

        # Track quantities
        train_inner_errors.append(meta_train_error / meta_bsz)
        train_inner_accuracies.append(meta_train_accuracy / meta_bsz)
        if iteration % test_interval == 0:
            valid_inner_errors.append(meta_valid_error / meta_bsz)
            valid_inner_accuracies.append(meta_valid_accuracy / meta_bsz)
            test_inner_errors.append(meta_test_error / meta_bsz)
            test_inner_accuracies.append(meta_test_accuracy / meta_bsz)

        # Average the accumulated gradients and optimize
        if iteration <= 20:
            for p in model.module._fc.parameters():
                p.grad.data.mul_(1.0 / meta_bsz).add_(p.data)
        else:
            for p in model.module.parameters():
                p.grad.data.mul_(1.0 / meta_bsz).add_(p.data)

        opt.step()


if __name__ == '__main__':
    main()
