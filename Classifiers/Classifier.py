import os
import timm
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.models
from torchvision.models import *
import datetime
from tqdm import tqdm
import sys
from efficientnet_pytorch import EfficientNet
from torchvision import transforms

# CMD run configuration
sys.path.append(r'/home/unknown/python_files/')
from Scripts.BuildDataset import *
from Scripts.ImageSegmentation import *


class Classifier:
    def __init__(self, model_name: str, model_path: str = None, num_classes: int = 2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = self.customModel(n_classes=num_classes)[model_name]  # build model, model is changeable
        print(f"[{self.device}] processor is used.")

        # This criterion computes the cross entropy loss between input logits and target.
        self.criterion = nn.CrossEntropyLoss()

        # load pre-trained model
        if model_path:
            checkpoint = torch.load(model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Using pre-loaded model: [{model_path}]")

        # eval() to set dropout and batch normalization layers to evaluation mode before running inference.
        # BatchNorm layers use running statistics, Dropout layers de-activated
        self.model.eval()

    def train(self, train_data: torch.utils.data.DataLoader, test_data: torch.utils.data.DataLoader,
              n_epochs: int, learning_rate: float, weight_decay: float, dataset_name: str) -> None:
        """
        train model with given custom parameters
        :param train_data: torch.utils.data.DataLoader (object).
        :param test_data: torch.utils.data.DataLoader (object).
        :param n_epochs: number of iteration of learning (int).
        :param learning_rate: learning rate curve (float).
        :param weight_decay: custom weight modifier (float).
        :return: None
        """
        continue_full_train = False
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dir_path = rootDir() + '/Stats' + f'/{dataset_name}_{self.model_name}_{current_time}'
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        # make train.txt file if not exists or clean it.
        with open(fr'{dir_path}/logs.txt', 'w+') as f:
            f.write('')

        # construct an optimizer object, hold the current state and will update the parameters based on gradients.
        if self.model_name == 'Efficientnet_B0':
            optimizer = torch.optim.SGD(self.model.module._fc.parameters(), lr=learning_rate)
        if self.model_name == 'Efficientnet_B7':
            optimizer = torch.optim.Adam(self.model.module._fc.parameters(), lr=learning_rate)
        if self.model_name == 'VGG16':
            optimizer = torch.optim.SGD(self.model.module.fc.parameters(), lr=learning_rate)
        if self.model_name == 'VGG19':
            optimizer = torch.optim.SGD(self.model.module.fc.parameters(), lr=learning_rate)
        if self.model_name == 'DenseNet121':
            optimizer = torch.optim.SGD(self.model.module.classifier.parameters(), lr=learning_rate)
        if self.model_name == 'DenseNet169':
            optimizer = torch.optim.SGD(self.model.module.classifier.parameters(), lr=learning_rate)
        if self.model_name == 'Inception_V3':
            optimizer = torch.optim.SGD(self.model.module.fc.parameters(), lr=learning_rate)
        if self.model_name == 'NasNet':
            optimizer = torch.optim.SGD(self.model.module.fc.parameters(), lr=learning_rate)
        if self.model_name == 'EfficientNet_B0_NS':
            optimizer = torch.optim.SGD(self.model.module.classifier.parameters(), lr=learning_rate)
        if self.model_name == 'IG_ResNext101_32x8D':
            optimizer = torch.optim.SGD(self.model.module.fc.parameters(), lr=learning_rate)
        if self.model_name == 'MobileNetv2_100':
            optimizer = torch.optim.SGD(self.model.module.classifier.parameters(), lr=learning_rate)
        if self.model_name == 'Xception':
            optimizer = torch.optim.SGD(self.model.module.fc.parameters(), lr=learning_rate)
        if self.model_name == 'RexNet_100':
            optimizer = torch.optim.SGD(self.model.module.head.fc.parameters(), lr=learning_rate)

        # Reduce learning rate when a metric has stopped improving (using optimizer object).
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3,
                                                               verbose=True)
        print("Training stage has started!", end='\r')
        best_acc = 0.0

        # train() tells your model that you are training the model.
        # This helps inform layers such as Dropout and BatchNorm
        self.model.train()
        for epoch in tqdm(range(n_epochs), position=0, desc="Iterations", leave=False, colour='RED', ncols=80,
                          total=n_epochs):

            if epoch > 20 and not continue_full_train:
                continue_full_train = True

                for param in self.model.module.parameters():
                    param.requires_grad = True

                optimizer = torch.optim.SGD(self.model.module.parameters(), lr=learning_rate)

            correct, total_labels, epoch_loss, loss = 0, 0, 0.0, 0

            # enumerate a BATCH size , amount of labels = num of dirs
            for i, (image, label) in tqdm(enumerate(train_data), position=1, desc="Train Stage", leave=False,
                                          colour='GREEN', ncols=80, total=len(train_data)):
                images = image.to(self.device)
                labels = label.to(self.device)  # returns a Tensor with the specified device and (optional) dtype.

                outputs = self.model(images)  # load tensors into model
                loss = self.criterion(outputs, labels)  # calculate loss

                optimizer.zero_grad()  # clear the gradients
                loss.backward()  # credit assignment
                optimizer.step()  # called once the gradients are computed using e.g. backward().

                epoch_loss += loss  # accumulate the loss value
                _, prediction = torch.max(outputs.data, 1)  # get max value each predicted cell.
                correct += (prediction == labels).sum().item()
                total_labels += labels.size(0)

            train_acc = 100 * correct / total_labels
            print(f'\n\nEpoch [{epoch + 1}/{n_epochs}]: train Loss: {loss} Train_Acc: {train_acc}', end='\n')

            # test accuracy at current epoch
            self.model.eval()  # start evaluation mode
            test_acc, test_loss = self.testAccuracy(test_data)

            # return to train mode
            self.model.train()

            print(
                f'\n\nEpoch [{epoch + 1}/{n_epochs}]: train Loss: {loss} Train_Acc: {train_acc}  test Loss: {test_loss} Test_Acc: {test_acc}',
                end='')

            with open(fr'{dir_path}/logs.txt', 'a') as f:
                f.write(f'Test|{epoch + 1}|{loss}|{train_acc}|{test_loss}|{test_acc}\n')

            # save the best accuracy model (pt file model).
            if test_acc > best_acc:
                best_acc = float("{:.2f}".format(test_acc))
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss, },
                    f'{dir_path}/{self.model_name}-{best_acc}-{current_time}.pt')

            scheduler.step(train_acc)  # update model weights

    def testAccuracy(self, data) -> tuple:

        with torch.no_grad():  # context-manager that disabled gradient calculation.
            correct, total_labels, loss = 0, 0, 0.0  # for all batch prediction
            for i, (images, labels) in tqdm((enumerate(data)), position=2, desc="Test Stage", leave=False,
                                            colour='BLUE', ncols=80, total=len(data)):
                # batch image section
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                _, prediction = torch.max(outputs.data, 1)
                loss = self.criterion(outputs, labels)
                correct += (prediction == labels).sum().item()
                total_labels += labels.size(0)

            return 100 * correct / total_labels, loss

    def customModel(self, n_classes: int = 2):
        def Efficientnet_B7_():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = EfficientNet.from_pretrained("efficientnet-b7")
            in_features = model._fc.in_features
            model._fc = nn.Linear(in_features, n_classes)

            # freeze all layers ( do not update gradients)
            for param in model.parameters():
                param.requires_grad = False

            # unfreeze last layer (update gradients)
            for param in model._fc.parameters():
                param.requires_grad = True

            model = torch.nn.DataParallel(model)
            model.to(device)
            return model

        def Efficientnet_B0_():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = EfficientNet.from_pretrained("efficientnet-b0")
            in_features = model._fc.in_features
            model._fc = nn.Linear(in_features, n_classes)

            # freeze all layers ( do not update gradients)
            for param in model.parameters():
                param.requires_grad = False

            # unfreeze last layer (update gradients)
            for param in model._fc.parameters():
                param.requires_grad = True

            model = torch.nn.DataParallel(model)
            model.to(device)
            return model

        def NasNet_():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = mnasnet0_5(weights=MNASNet0_5_Weights.DEFAULT)
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, n_classes)

            # freeze all layers ( do not update gradients)
            for param in model.parameters():
                param.requires_grad = False

            # unfreeze last layer (update gradients)
            for param in model.classifier[-1].parameters():
                param.requires_grad = True

            model = torch.nn.DataParallel(model)
            model.to(device)
            return model

        def VGG16_():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = vgg16(weights=VGG16_Weights.DEFAULT)
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, n_classes)

            # freeze all layers ( do not update gradients)
            for param in model.parameters():
                param.requires_grad = False

            # unfreeze last layer (update gradients)
            for param in model.classifier[-1].parameters():
                param.requires_grad = True

            model = torch.nn.DataParallel(model)
            model.to(device)
            return model

        def VGG19_():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = vgg19(weights=VGG19_Weights.DEFAULT)
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, n_classes)

            # freeze all layers ( do not update gradients)
            for param in model.parameters():
                param.requires_grad = False

            # unfreeze last layer (update gradients)
            for param in model.classifier[-1].parameters():
                param.requires_grad = True

            model = torch.nn.DataParallel(model)
            model.to(device)
            return model

        def Inception_V3_():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, n_classes)

            # freeze all layers ( do not update gradients)
            for param in model.parameters():
                param.requires_grad = False

            # unfreeze last layer (update gradients)
            for param in model.fc.parameters():
                param.requires_grad = True

            model = torch.nn.DataParallel(model)
            model.to(device)
            return model

        def DenseNet121_():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = densenet121(weights=DenseNet121_Weights.DEFAULT)
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, n_classes)

            # freeze all layers ( do not update gradients)
            for param in model.parameters():
                param.requires_grad = False

            # unfreeze last layer (update gradients)
            for param in model.classifier.parameters():
                param.requires_grad = True

            model = torch.nn.DataParallel(model)
            model.to(device)
            return model

        def DenseNet169_():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = densenet169(weights=DenseNet169_Weights.DEFAULT)
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, n_classes)

            # freeze all layers ( do not update gradients)
            for param in model.parameters():
                param.requires_grad = False

            # unfreeze last layer (update gradients)
            for param in model.classifier.parameters():
                param.requires_grad = True

            model = torch.nn.DataParallel(model)
            model.to(device)
            return model

        def EfficientNet_B0_NS():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = timm.create_model('tf_efficientnet_b0_ns', pretrained=True)
            in_features = model.classifier.in_features
            model.classifier= nn.Linear(in_features, n_classes)

            # freeze all layers ( do not update gradients)
            for param in model.parameters():
                param.requires_grad = False

            # unfreeze last layer (update gradients)
            for param in model.classifier.parameters():
                param.requires_grad = True

            model = torch.nn.DataParallel(model)
            model.to(device)
            return model

        def IG_ResNext101_32x8D():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = timm.create_model('ig_resnext101_32x8d', pretrained=True)
            in_features = model.fc.in_features
            model.fc= nn.Linear(in_features, n_classes)

            # freeze all layers ( do not update gradients)
            for param in model.parameters():
                param.requires_grad = False

            # unfreeze last layer (update gradients)
            for param in model.fc.parameters():
                param.requires_grad = True

            model = torch.nn.DataParallel(model)
            model.to(device)
            return model

        def Xception():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = timm.create_model('xception', pretrained=True)
            in_features = model.fc.in_features
            model.fc= nn.Linear(in_features, n_classes)

            # freeze all layers ( do not update gradients)
            for param in model.parameters():
                param.requires_grad = False

            # unfreeze last layer (update gradients)
            for param in model.fc.parameters():
                param.requires_grad = True

            model = torch.nn.DataParallel(model)
            model.to(device)
            return model

        def MobileNetv2_100():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = timm.create_model('mobilenetv2_100', pretrained=True)
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, n_classes)

            # freeze all layers ( do not update gradients)
            for param in model.parameters():
                param.requires_grad = False

            # unfreeze last layer (update gradients)
            for param in model.classifier.parameters():
                param.requires_grad = True

            model = torch.nn.DataParallel(model)
            model.to(device)
            return model

        def RexNet_100():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = timm.create_model('rexnet_100', pretrained=True)
            model.head.fc = nn.Linear(1280, n_classes)

            # freeze all layers ( do not update gradients)
            for param in model.parameters():
                param.requires_grad = False

            # unfreeze last layer (update gradients)
            for param in model.head.fc.parameters():
                param.requires_grad = True

            model = torch.nn.DataParallel(model)
            model.to(device)
            return model

        return {
            'Efficientnet_B0': Efficientnet_B0_(),
            'Efficientnet_B7': Efficientnet_B7_(),
            'NasNet': NasNet_(),
            'VGG16': VGG16_(),
            'VGG19': VGG19_(),
            'Inception_V3': Inception_V3_(),
            'DenseNet121': DenseNet121_(),
            'DenseNet169': DenseNet169_(),
            'EfficientNet_B0_NS': EfficientNet_B0_NS(),
            'IG_ResNext101_32x8D': IG_ResNext101_32x8D(),
            'MobileNetv2_100': MobileNetv2_100(),
            'Xception': Xception(),
            'RexNet_100': RexNet_100()
        }


clf = Classifier(model_name='Efficientnet_B7', model_path=None, num_classes=4)
clf.train(TorchDataLoader('KHATT_COMBINED_PARAGRAPH', n_classes=4, kind='Full').train['age'],
          TorchDataLoader('KHATT_COMBINED_PARAGRAPH', n_classes=4, kind='Full').test['age'],
          n_epochs=50,
          learning_rate=0.005,
          weight_decay=1e-2,
          dataset_name='KHATT_COMBINED_Paragraph_Age_4_Classes_New_Pipeline___1')

clf.train(TorchDataLoader('KHATT_COMBINED_PARAGRAPH', n_classes=4, kind='Full').train['age'],
          TorchDataLoader('KHATT_COMBINED_PARAGRAPH', n_classes=4, kind='Full').test['age'],
          n_epochs=50,
          learning_rate=0.005,
          weight_decay=1e-2,
          dataset_name='KHATT_COMBINED_Paragraph_Age_4_Classes_New_Pipeline___2')

clf.train(TorchDataLoader('KHATT_COMBINED_PARAGRAPH', n_classes=4, kind='Full').train['age'],
          TorchDataLoader('KHATT_COMBINED_PARAGRAPH', n_classes=4, kind='Full').test['age'],
          n_epochs=50,
          learning_rate=0.005,
          weight_decay=1e-2,
          dataset_name='KHATT_COMBINED_Paragraph_Age_4_Classes_New_Pipeline___3')

clf.train(TorchDataLoader('KHATT_COMBINED_PARAGRAPH', n_classes=4, kind='Full').train['age'],
          TorchDataLoader('KHATT_COMBINED_PARAGRAPH', n_classes=4, kind='Full').test['age'],
          n_epochs=50,
          learning_rate=0.005,
          weight_decay=1e-2,
          dataset_name='KHATT_COMBINED_Paragraph_Age_4_Classes_New_Pipeline___4')

clf.train(TorchDataLoader('KHATT_COMBINED_PARAGRAPH', n_classes=4, kind='Full').train['age'],
          TorchDataLoader('KHATT_COMBINED_PARAGRAPH', n_classes=4, kind='Full').test['age'],
          n_epochs=50,
          learning_rate=0.005,
          weight_decay=1e-2,
          dataset_name='KHATT_COMBINED_Paragraph_Age_4_Classes_New_Pipeline___5')

clf.train(TorchDataLoader('KHATT_COMBINED_PARAGRAPH', n_classes=4, kind='Full').train['age'],
          TorchDataLoader('KHATT_COMBINED_PARAGRAPH', n_classes=4, kind='Full').test['age'],
          n_epochs=50,
          learning_rate=0.005,
          weight_decay=1e-2,
          dataset_name='KHATT_COMBINED_Paragraph_Age_4_Classes_New_Pipeline___6')