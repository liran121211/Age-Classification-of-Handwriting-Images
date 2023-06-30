import os
import random
import h5py
import sys
import torch
import numpy as np
import torchvision
from PIL import Image
from abc import abstractmethod, ABC
from torch.utils.data import DataLoader
from torchvision import datasets

# CMD run configuration
sys.path.append(r'/home/unknown/python_files/')
from Scripts.Enums import *


def to_negative(x: Image) -> Image:
    """
    Transform image to negative form.
    :param x: Image (object)
    :return: Negative image (object).
    """
    im_np = np.asarray(x)
    im_np = 255 - im_np
    im_pil = Image.fromarray(im_np)
    return im_pil


def rootDir():
    """
    Get current project absolute path.
    :return: absolute system path of project.
    :rtype: (String)
    """
    abs_path = os.path.abspath('')
    project_dir = os.path.dirname(abs_path)
    return project_dir


def transform_func(pil_img: Image):
    return torchvision.transforms.Compose([
        # Resize the input image to the given size.
        # torchvision.transforms.Resize((1280, 1280)),
        # torchvision.transforms.Lambda(to_negative),

        # Crops the given image at the center.
        # torchvision.transforms.RandomAdjustSharpness(0.8),

        # torchvision.transforms.RandomRotation(20),
        # torchvision.transforms.RandomHorizontalFlip(),
        # torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        # torchvision.transforms.RandomEqualize(),
        # torchvision.transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
        # torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        torchvision.transforms.RandomRotation(degrees=30),
        torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomResizedCrop(size=(1000, 1000), scale=(0.08, 1.0), ratio=(0.75, 1.33)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
    ])(pil_img)


class Dataset_H5(torch.utils.data.Dataset):
    def __init__(self, in_file, transform=None):
        super(Dataset_H5, self).__init__()

        self.file = h5py.File(in_file, 'r')
        self.transform = transform  # pass as object function

    def __getitem__(self, index):
        x = self.file['images'][index, ...]
        y = self.file['labels'][index, ...]

        # Preprocessing each image
        if self.transform is not None:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return self.file['images'].shape[0]


class Builder(ABC):
    """
    The Builder interface specifies methods for creating the different parts of the Product objects.
    """

    @property
    @abstractmethod
    def loader(self) -> None:
        pass

    @abstractmethod
    def train(self) -> DataLoader:
        pass

    @abstractmethod
    def test(self) -> DataLoader:
        pass

    @abstractmethod
    def val(self) -> DataLoader:
        pass


class QUWI:
    def __init__(self):
        self._train_ara_data_dir = rootDir() + r'/Datasets/QUWI/arabic/train/patches'
        self._train_eng_data_dir = rootDir() + r'/Datasets/QUWI/english/train/patches'
        self._val_ara_data_dir = rootDir() + r'/Datasets/QUWI/arabic/test/patches'
        self._val_eng_data_dir = rootDir() + r'/Datasets/QUWI/english/test/patches'
        self._test_ara_data_dir = rootDir() + r'/Datasets/QUWI/arabic/test/patches'
        self._test_eng_data_dir = rootDir() + r'/Datasets/QUWI/english/test/patches'
        self._batch_size = Pytorch.NORMAL_BATCH.value
        self._transform_func = transform_func

    def train(self):
        # load images files and apply transform pipeline
        # train_ara_data = datasets.ImageFolder(self._train_ara_data_dir, self._transform_func)
        train_eng_data = datasets.ImageFolder(self._train_eng_data_dir, self._transform_func)
        # combined_train_datasets = torch.utils.data.ConcatDataset([train_ara_data, train_eng_data])

        return torch.utils.data.DataLoader(dataset=train_eng_data, batch_size=self._batch_size, shuffle=True)

    def test(self):
        # load images files and apply transform pipeline
        # test_ara_data = datasets.ImageFolder(self._test_ara_data_dir, self._transform_func)
        test_eng_data = datasets.ImageFolder(self._test_eng_data_dir, self._transform_func)

        # combined_test_datasets = torch.utils.data.ConcatDataset([test_ara_data, test_eng_data])
        return torch.utils.data.DataLoader(dataset=test_eng_data, batch_size=self._batch_size, shuffle=True)

    def val(self):
        # load images files and apply transform pipeline
        # val_ara_data = datasets.ImageFolder(self._val_ara_data_dir, self._transform_func)
        val_eng_data = datasets.ImageFolder(self._val_eng_data_dir, self._transform_func)

        # combined_val_datasets = torch.utils.data.ConcatDataset([val_ara_data, val_eng_data])
        return torch.utils.data.DataLoader(dataset=val_eng_data, batch_size=self._batch_size, shuffle=True)


class HHD_Gender:
    def __init__(self, kind):
        if kind == 'Full':
            self._train_data_dir = rootDir() + r'/Datasets/HHD_Gender/Train/Full_Size/'
            self._val_data_dir = rootDir() + r'/Datasets/HHD_Gender/Validation/Full_Size/'
            self._test_data_dir = rootDir() + r'/Datasets/HHD_Gender/Test/Full_Size/'
            self._batch_size = Pytorch.DEFAULT_BATCH.value
        else:
            self._train_data_dir = rootDir() + r'/Datasets/HHD_Gender/Train/Patches/'
            self._val_data_dir = rootDir() + r'/Datasets/HHD_Gender/Validation/Patches/'
            self._test_data_dir = rootDir() + r'/Datasets/HHD_Gender/Test/Patches/'
            self._batch_size = Pytorch.NORMAL_BATCH.value

        self._transform_func = transform_func

    def train(self):
        # load images files and apply transform pipeline
        train_data = datasets.ImageFolder(self._train_data_dir, self._transform_func)
        return torch.utils.data.DataLoader(dataset=train_data, batch_size=self._batch_size, shuffle=True)

    def test(self):
        # load images files and apply transform pipeline
        test_data = datasets.ImageFolder(self._test_data_dir, self._transform_func)
        return torch.utils.data.DataLoader(dataset=test_data, batch_size=self._batch_size, shuffle=True)

    def val(self):
        # load images files and apply transform pipeline
        val_data = datasets.ImageFolder(self._val_data_dir, self._transform_func)
        return torch.utils.data.DataLoader(dataset=val_data, batch_size=self._batch_size, shuffle=True)


class HHD_Age:
    def __init__(self, kind, n_classes):
        if kind == 'Full':
            self._train_data_dir = rootDir() + fr'/Datasets/HHD_Age/Train/Full_Size/{n_classes}_Classes'
            self._val_data_dir = rootDir() + fr'/Datasets/HHD_Age/Validation/Full_Size/{n_classes}_Classes'
            self._test_data_dir = rootDir() + fr'/Datasets/HHD_Age/Test/Full_Size/{n_classes}_Classes'
            self._batch_size = Pytorch.NORMAL_BATCH.value
        else:
            self._train_data_dir = rootDir() + fr'/Datasets/HHD_Age/Train/Segments/{n_classes}_Classes'
            self._val_data_dir = rootDir() + fr'/Datasets/HHD_Age/Validation/Segments/{n_classes}_Classes'
            self._test_data_dir = rootDir() + fr'/Datasets/HHD_Age/Test/Segments/{n_classes}_Classes'
            self._batch_size = Pytorch.NORMAL_BATCH.value

        self._transform_func = transform_func

    def train(self):
        # load images files and apply transform pipeline
        train_data = datasets.ImageFolder(self._train_data_dir, self._transform_func)
        return torch.utils.data.DataLoader(dataset=train_data, batch_size=self._batch_size, shuffle=True)

    def test(self):
        # load images files and apply transform pipeline
        test_data = datasets.ImageFolder(self._test_data_dir, self._transform_func)
        return torch.utils.data.DataLoader(dataset=test_data, batch_size=self._batch_size, shuffle=True)

    def val(self):
        # load images files and apply transform pipeline
        val_data = datasets.ImageFolder(self._val_data_dir, self._transform_func)
        return torch.utils.data.DataLoader(dataset=val_data, batch_size=self._batch_size, shuffle=True)


class KHATT_Fixed_LineImages:
    def __init__(self, n_classes):
        self._train_age_data_dir = rootDir() + fr'/Datasets/KHATT/LineImages_v1.0/FixedTextLineImages/Train/Age/{n_classes}_Classes'
        self._val_age_data_dir = rootDir() + fr'/Datasets/KHATT/LineImages_v1.0/FixedTextLineImages/Validate/Age/{n_classes}_Classes'
        self._test_age_data_dir = rootDir() + fr'/Datasets/KHATT/LineImages_v1.0/FixedTextLineImages/Test/Age/{n_classes}_Classes'
        self._train_gender_data_dir = rootDir() + r'/Datasets/KHATT/LineImages_v1.0/FixedTextLineImages/Train/Gender'
        self._val_gender_data_dir = rootDir() + r'/Datasets/KHATT/LineImages_v1.0/FixedTextLineImages/Validate/Gender'
        self._test_gender_data_dir = rootDir() + r'/Datasets/KHATT/LineImages_v1.0/FixedTextLineImages/Test/Gender'
        self._batch_size = Pytorch.DEFAULT_BATCH.value
        self._transform_func = transform_func

    def train(self):
        # load images files and apply transform pipeline
        def age():
            age_data = datasets.ImageFolder(self._train_age_data_dir, self._transform_func)
            return torch.utils.data.DataLoader(dataset=age_data, batch_size=self._batch_size, shuffle=True)

        def gender():
            gender_data = datasets.ImageFolder(self._train_gender_data_dir, self._transform_func)
            return torch.utils.data.DataLoader(dataset=gender_data, batch_size=self._batch_size, shuffle=True)

        return {
            'age': age,
            'gender': gender
        }

    def test(self):
        # load images files and apply transform pipeline
        def age():
            age_data = datasets.ImageFolder(self._test_age_data_dir, self._transform_func)
            return torch.utils.data.DataLoader(dataset=age_data, batch_size=self._batch_size, shuffle=True)

        def gender():
            gender_data = datasets.ImageFolder(self._test_gender_data_dir, self._transform_func)
            return torch.utils.data.DataLoader(dataset=gender_data, batch_size=self._batch_size, shuffle=True)

        return {
            'age': age,
            'gender': gender
        }

    def val(self):
        # load images files and apply transform pipeline
        def age():
            age_data = datasets.ImageFolder(self._val_age_data_dir, self._transform_func)
            return torch.utils.data.DataLoader(dataset=age_data, batch_size=self._batch_size, shuffle=True)

        def gender():
            gender_data = datasets.ImageFolder(self._val_gender_data_dir, self._transform_func)
            return torch.utils.data.DataLoader(dataset=gender_data, batch_size=self._batch_size, shuffle=True)

        return {
            'age': age,
            'gender': gender
        }


class KHATT_Unique_LineImages:
    def __init__(self, n_classes):
        self._train_age_data_dir = rootDir() + fr'/Datasets/KHATT/LineImages_v1.0/UniqueTextLineImages/Train/Age/{n_classes}_Classes'
        self._val_age_data_dir = rootDir() + fr'/Datasets/KHATT/LineImages_v1.0/UniqueTextLineImages/Validate/Age/{n_classes}_Classes'
        self._test_age_data_dir = rootDir() + fr'/Datasets/KHATT/LineImages_v1.0/UniqueTextLineImages/Test/Age/{n_classes}_Classes'
        self._train_gender_data_dir = rootDir() + r'/Datasets/KHATT/LineImages_v1.0/UniqueTextLineImages/Train/Gender'
        self._val_gender_data_dir = rootDir() + r'/Datasets/KHATT/LineImages_v1.0/UniqueTextLineImages/Validate/Gender'
        self._test_gender_data_dir = rootDir() + r'/Datasets/KHATT/LineImages_v1.0/UniqueTextLineImages/Test/Gender'
        self._batch_size = Pytorch.DEFAULT_BATCH.value
        self._transform_func = transform_func

    def train(self):
        # load images files and apply transform pipeline
        def age():
            age_data = datasets.ImageFolder(self._train_age_data_dir, self._transform_func)
            return torch.utils.data.DataLoader(dataset=age_data, batch_size=self._batch_size, shuffle=True)

        def gender():
            gender_data = datasets.ImageFolder(self._train_gender_data_dir, self._transform_func)
            return torch.utils.data.DataLoader(dataset=gender_data, batch_size=self._batch_size, shuffle=True)

        return {
            'age': age,
            'gender': gender
        }

    def test(self):
        # load images files and apply transform pipeline
        def age():
            age_data = datasets.ImageFolder(self._test_age_data_dir, self._transform_func)
            return torch.utils.data.DataLoader(dataset=age_data, batch_size=self._batch_size, shuffle=True)

        def gender():
            gender_data = datasets.ImageFolder(self._test_gender_data_dir, self._transform_func)
            return torch.utils.data.DataLoader(dataset=gender_data, batch_size=self._batch_size, shuffle=True)

        return {
            'age': age,
            'gender': gender
        }

    def val(self):
        # load images files and apply transform pipeline
        def age():
            age_data = datasets.ImageFolder(self._val_age_data_dir, self._transform_func)
            return torch.utils.data.DataLoader(dataset=age_data, batch_size=self._batch_size, shuffle=True)

        def gender():
            gender_data = datasets.ImageFolder(self._val_gender_data_dir, self._transform_func)
            return torch.utils.data.DataLoader(dataset=gender_data, batch_size=self._batch_size, shuffle=True)

        return {
            'age': age,
            'gender': gender
        }


class KHATT_Fixed_ParagraphImages:
    def __init__(self, n_classes):
        self._train_age_data_dir = rootDir() + fr'/Datasets/KHATT/ParagraphImages_v1.0/FixedTextParagraphs/Train/Age/{n_classes}_Classes'
        self._val_age_data_dir = rootDir() + fr'/Datasets/KHATT/ParagraphImages_v1.0/FixedTextParagraphs/Validate/Age/{n_classes}_Classes'
        self._test_age_data_dir = rootDir() + fr'/Datasets/KHATT/ParagraphImages_v1.0/FixedTextParagraphs/Test/Age/{n_classes}_Classes'
        self._train_gender_data_dir = rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/FixedTextParagraphs/Train/Gender'
        self._val_gender_data_dir = rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/FixedTextParagraphs/Validate/Gender'
        self._test_gender_data_dir = rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/FixedTextParagraphs/Test/Gender'

        self._batch_size = Pytorch.NORMAL_BATCH.value
        self._transform_func = transform_func

    def train(self, n_classes):
        # load images files and apply transform pipeline
        def age():
            age_data = datasets.ImageFolder(self._train_age_data_dir, self._transform_func)
            return torch.utils.data.DataLoader(dataset=age_data, batch_size=self._batch_size, shuffle=True)

        def gender():
            gender_data = datasets.ImageFolder(self._train_gender_data_dir, self._transform_func)
            return torch.utils.data.DataLoader(dataset=gender_data, batch_size=self._batch_size, shuffle=True)

        return {
            'age': age,
            'gender': gender
        }

    def test(self):
        # load images files and apply transform pipeline
        def age():
            age_data = datasets.ImageFolder(self._test_age_data_dir, self._transform_func)
            return torch.utils.data.DataLoader(dataset=age_data, batch_size=self._batch_size, shuffle=True)

        def gender():
            gender_data = datasets.ImageFolder(self._test_gender_data_dir, self._transform_func)
            return torch.utils.data.DataLoader(dataset=gender_data, batch_size=self._batch_size, shuffle=True)

        return {
            'age': age,
            'gender': gender
        }

    def val(self):
        # load images files and apply transform pipeline
        def age():
            age_data = datasets.ImageFolder(self._val_age_data_dir, self._transform_func)
            return torch.utils.data.DataLoader(dataset=age_data, batch_size=self._batch_size, shuffle=True)

        def gender():
            gender_data = datasets.ImageFolder(self._val_gender_data_dir, self._transform_func)
            return torch.utils.data.DataLoader(dataset=gender_data, batch_size=self._batch_size, shuffle=True)

        return {
            'age': age,
            'gender': gender
        }


class KHATT_Unique_ParagraphImages:
    def __init__(self, n_classes):
        self._train_age_data_dir = rootDir() + fr'/Datasets/KHATT/ParagraphImages_v1.0/UniqueTextParagraphs/Train/Age/{n_classes}_Classes'
        self._val_age_data_dir = rootDir() + fr'/Datasets/KHATT/ParagraphImages_v1.0/UniqueTextParagraphs/Validate/Age/{n_classes}_Classes'
        self._test_age_data_dir = rootDir() + fr'/Datasets/KHATT/ParagraphImages_v1.0/UniqueTextParagraphs/Test/Age/{n_classes}_Classes'
        self._train_gender_data_dir = rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/UniqueTextParagraphs/Train/Gender'
        self._val_gender_data_dir = rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/UniqueTextParagraphs/Validate/Gender'
        self._test_gender_data_dir = rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/UniqueTextParagraphs/Test/Gender'

        self._batch_size = Pytorch.NORMAL_BATCH.value
        self._transform_func = transform_func

    def train(self):
        # load images files and apply transform pipeline
        def age():
            age_data = datasets.ImageFolder(self._train_age_data_dir, self._transform_func)
            return torch.utils.data.DataLoader(dataset=age_data, batch_size=self._batch_size, shuffle=True)

        def gender():
            gender_data = datasets.ImageFolder(self._train_gender_data_dir, self._transform_func)
            return torch.utils.data.DataLoader(dataset=gender_data, batch_size=self._batch_size, shuffle=True)

        return {
            'age': age,
            'gender': gender
        }

    def test(self):
        # load images files and apply transform pipeline
        def age():
            age_data = datasets.ImageFolder(self._test_age_data_dir, self._transform_func)
            return torch.utils.data.DataLoader(dataset=age_data, batch_size=self._batch_size, shuffle=True)

        def gender():
            gender_data = datasets.ImageFolder(self._test_gender_data_dir, self._transform_func)
            return torch.utils.data.DataLoader(dataset=gender_data, batch_size=self._batch_size, shuffle=True)

        return {
            'age': age,
            'gender': gender
        }

    def val(self):
        # load images files and apply transform pipeline
        def age():
            age_data = datasets.ImageFolder(self._val_age_data_dir, self._transform_func)
            return torch.utils.data.DataLoader(dataset=age_data, batch_size=self._batch_size, shuffle=True)

        def gender():
            gender_data = datasets.ImageFolder(self._val_gender_data_dir, self._transform_func)
            return torch.utils.data.DataLoader(dataset=gender_data, batch_size=self._batch_size, shuffle=True)

        return {
            'age': age,
            'gender': gender
        }


class KHATT_Combined_ParagraphImages:
    def __init__(self, n_classes):
        self._unique_train_age_data_dir = rootDir() + fr'/Datasets/KHATT/ParagraphImages_v1.0/UniqueTextParagraphs/Train/Age/{n_classes}_Classes'
        self._unique_val_age_data_dir = rootDir() + fr'/Datasets/KHATT/ParagraphImages_v1.0/UniqueTextParagraphs/Validate/Age/{n_classes}_Classes'
        self._unique_test_age_data_dir = rootDir() + fr'/Datasets/KHATT/ParagraphImages_v1.0/UniqueTextParagraphs/Test/Age/{n_classes}_Classes'

        self._fixed_train_age_data_dir = rootDir() + fr'/Datasets/KHATT/ParagraphImages_v1.0/FixedTextParagraphs/Train/Age/{n_classes}_Classes'
        self._fixed_val_age_data_dir = rootDir() + fr'/Datasets/KHATT/ParagraphImages_v1.0/FixedTextParagraphs/Validate/Age/{n_classes}_Classes'
        self._fixed_test_age_data_dir = rootDir() + fr'/Datasets/KHATT/ParagraphImages_v1.0/FixedTextParagraphs/Test/Age/{n_classes}_Classes'

        self._unique_train_gender_data_dir = rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/UniqueTextParagraphs/Train/Gender'
        self._unique_val_gender_data_dir = rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/UniqueTextParagraphs/Validate/Gender'
        self._unique_test_gender_data_dir = rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/UniqueTextParagraphs/Test/Gender'

        self._fixed_train_gender_data_dir = rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/FixedTextParagraphs/Train/Gender'
        self._fixed_val_gender_data_dir = rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/FixedTextParagraphs/Validate/Gender'
        self._fixed_test_gender_data_dir = rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/FixedTextParagraphs/Test/Gender'

        self._batch_size = Pytorch.NORMAL_BATCH.value
        self._transform_func = transform_func

    def train(self):
        # load images files and apply transform pipeline
        def age():
            unique_age_data = datasets.ImageFolder(self._unique_train_age_data_dir, self._transform_func)
            fixed_age_data = datasets.ImageFolder(self._fixed_train_age_data_dir, self._transform_func)
            combined_train_datasets = torch.utils.data.ConcatDataset([unique_age_data, fixed_age_data])
            return torch.utils.data.DataLoader(dataset=combined_train_datasets, batch_size=self._batch_size,
                                               shuffle=True)

        def gender():
            unique_gender_data = datasets.ImageFolder(self._unique_train_gender_data_dir, self._transform_func)
            fixed_gender_data = datasets.ImageFolder(self._fixed_train_gender_data_dir, self._transform_func)
            combined_train_datasets = torch.utils.data.ConcatDataset([unique_gender_data, fixed_gender_data])
            return torch.utils.data.DataLoader(dataset=combined_train_datasets, batch_size=self._batch_size,
                                               shuffle=True)

        return {
            'age': age,
            'gender': gender
        }

    def test(self):
        # load images files and apply transform pipeline
        def age():
            unique_age_data_1 = datasets.ImageFolder(self._unique_val_age_data_dir, self._transform_func)
            unique_age_data_2 = datasets.ImageFolder(self._unique_test_age_data_dir, self._transform_func)
            fixed_age_data_3 = datasets.ImageFolder(self._fixed_val_age_data_dir, self._transform_func)
            fixed_age_data_4 = datasets.ImageFolder(self._fixed_test_age_data_dir, self._transform_func)
            combined_datasets = torch.utils.data.ConcatDataset([unique_age_data_1, unique_age_data_2, fixed_age_data_3, fixed_age_data_4])
            return torch.utils.data.DataLoader(dataset=combined_datasets, batch_size=self._batch_size,
                                               shuffle=True)

        def gender():
            unique_gender_data = datasets.ImageFolder(self._unique_val_gender_data_dir, self._transform_func)
            fixed_gender_data = datasets.ImageFolder(self._fixed_val_gender_data_dir, self._transform_func)
            combined_validation_datasets = torch.utils.data.ConcatDataset([unique_gender_data, fixed_gender_data])
            return torch.utils.data.DataLoader(dataset=combined_validation_datasets, batch_size=self._batch_size,
                                               shuffle=True)

        return {
            'age': age,
            'gender': gender
        }

    def val(self):
        # load images files and apply transform pipeline
        def age():
            unique_age_data = datasets.ImageFolder(self._unique_test_age_data_dir, self._transform_func)
            fixed_age_data = datasets.ImageFolder(self._fixed_test_age_data_dir, self._transform_func)
            combined_test_datasets = torch.utils.data.ConcatDataset([unique_age_data, fixed_age_data])
            return torch.utils.data.DataLoader(dataset=combined_test_datasets, batch_size=self._batch_size,
                                               shuffle=True)

        def gender():
            unique_gender_data = datasets.ImageFolder(self._unique_test_gender_data_dir, self._transform_func)
            fixed_gender_data = datasets.ImageFolder(self._fixed_test_gender_data_dir, self._transform_func)
            combined_test_datasets = torch.utils.data.ConcatDataset([unique_gender_data, fixed_gender_data])
            return torch.utils.data.DataLoader(dataset=combined_test_datasets, batch_size=self._batch_size,
                                               shuffle=True)

        return {
            'age': age,
            'gender': gender
        }


class KHATT_Combined_LineImages:
    def __init__(self, n_classes):
        self._unique_train_age_data_dir = rootDir() + fr'/Datasets/KHATT/LineImages_v1.0/UniqueTextLineImages/Train/Age/{n_classes}_Classes'
        self._unique_val_age_data_dir = rootDir() + fr'/Datasets/KHATT/LineImages_v1.0/UniqueTextLineImages/Validate/Age/{n_classes}_Classes'
        self._unique_test_age_data_dir = rootDir() + fr'/Datasets/KHATT/LineImages_v1.0/UniqueTextLineImages/Test/Age/{n_classes}_Classes'

        self._fixed_train_age_data_dir = rootDir() + fr'/Datasets/KHATT/LineImages_v1.0/FixedTextLineImages/Train/Age/{n_classes}_Classes'
        self._fixed_val_age_data_dir = rootDir() + fr'/Datasets/KHATT/LineImages_v1.0/FixedTextLineImages/Validate/Age/{n_classes}_Classes'
        self._fixed_test_age_data_dir = rootDir() + fr'/Datasets/KHATT/LineImages_v1.0/FixedTextLineImages/Test/Age/{n_classes}_Classes'

        self._unique_train_gender_data_dir = rootDir() + r'/Datasets/KHATT/LineImages_v1.0/UniqueTextLineImages/Train/Gender'
        self._unique_val_gender_data_dir = rootDir() + r'/Datasets/KHATT/LineImages_v1.0/UniqueTextLineImages/Validate/Gender'
        self._unique_test_gender_data_dir = rootDir() + r'/Datasets/KHATT/LineImages_v1.0/UniqueTextLineImages/Test/Gender'

        self._fixed_train_gender_data_dir = rootDir() + r'/Datasets/KHATT/LineImages_v1.0/FixedTextLineImages/Train/Gender'
        self._fixed_val_gender_data_dir = rootDir() + r'/Datasets/KHATT/LineImages_v1.0/FixedTextLineImages/Validate/Gender'
        self._fixed_test_gender_data_dir = rootDir() + r'/Datasets/KHATT/LineImages_v1.0/FixedTextLineImages/Test/Gender'

        self._batch_size = Pytorch.NORMAL_BATCH.value
        self._transform_func = transform_func

    def train(self):
        # load images files and apply transform pipeline
        def age():
            unique_age_data = datasets.ImageFolder(self._unique_train_age_data_dir, self._transform_func)
            fixed_age_data = datasets.ImageFolder(self._fixed_train_age_data_dir, self._transform_func)
            combined_train_datasets = torch.utils.data.ConcatDataset([unique_age_data, fixed_age_data])
            return torch.utils.data.DataLoader(dataset=combined_train_datasets, batch_size=self._batch_size,
                                               shuffle=True)

        def gender():
            unique_gender_data = datasets.ImageFolder(self._unique_train_gender_data_dir, self._transform_func)
            fixed_gender_data = datasets.ImageFolder(self._fixed_train_gender_data_dir, self._transform_func)
            combined_train_datasets = torch.utils.data.ConcatDataset([unique_gender_data, fixed_gender_data])
            return torch.utils.data.DataLoader(dataset=combined_train_datasets, batch_size=self._batch_size,
                                               shuffle=True)

        return {
            'age': age,
            'gender': gender
        }

    def test(self):
        # load images files and apply transform pipeline
        def age():
            unique_age_data = datasets.ImageFolder(self._unique_val_age_data_dir, self._transform_func)
            fixed_age_data = datasets.ImageFolder(self._fixed_val_age_data_dir, self._transform_func)
            combined_validation_datasets = torch.utils.data.ConcatDataset([unique_age_data, fixed_age_data])
            return torch.utils.data.DataLoader(dataset=combined_validation_datasets, batch_size=self._batch_size,
                                               shuffle=True)

        def gender():
            unique_gender_data = datasets.ImageFolder(self._unique_val_gender_data_dir, self._transform_func)
            fixed_gender_data = datasets.ImageFolder(self._fixed_val_gender_data_dir, self._transform_func)
            combined_validation_datasets = torch.utils.data.ConcatDataset([unique_gender_data, fixed_gender_data])
            return torch.utils.data.DataLoader(dataset=combined_validation_datasets, batch_size=self._batch_size,
                                               shuffle=True)

        return {
            'age': age,
            'gender': gender
        }

    def val(self):
        # load images files and apply transform pipeline
        def age():
            unique_age_data = datasets.ImageFolder(self._unique_test_age_data_dir, self._transform_func)
            fixed_age_data = datasets.ImageFolder(self._fixed_test_age_data_dir, self._transform_func)
            combined_test_datasets = torch.utils.data.ConcatDataset([unique_age_data, fixed_age_data])
            return torch.utils.data.DataLoader(dataset=combined_test_datasets, batch_size=self._batch_size,
                                               shuffle=True)

        def gender():
            unique_gender_data = datasets.ImageFolder(self._unique_test_gender_data_dir, self._transform_func)
            fixed_gender_data = datasets.ImageFolder(self._fixed_test_gender_data_dir, self._transform_func)
            combined_test_datasets = torch.utils.data.ConcatDataset([unique_gender_data, fixed_gender_data])
            return torch.utils.data.DataLoader(dataset=combined_test_datasets, batch_size=self._batch_size,
                                               shuffle=True)

        return {
            'age': age,
            'gender': gender
        }


class MixedLanguages:
    def __init__(self):
        self._train_data_dir = rootDir() + r'/Datasets/Mixed/train/'
        self._val_data_dir = rootDir() + r'/Datasets/Mixed/val/'
        self._test_data_dir = rootDir() + r'/Datasets/Mixed/test/'
        self._batch_size = Pytorch.DEFAULT_BATCH.value
        self._transform_func = transform_func

    def train(self):
        # load images files and apply transform pipeline
        train_data = datasets.ImageFolder(self._train_data_dir, self._transform_func)
        return torch.utils.data.DataLoader(dataset=train_data, batch_size=self._batch_size, shuffle=True)

    def test(self):
        # load images files and apply transform pipeline
        test_data = datasets.ImageFolder(self._test_data_dir, self._transform_func)
        return torch.utils.data.DataLoader(dataset=test_data, batch_size=self._batch_size, shuffle=True)

    def val(self):
        # load images files and apply transform pipeline
        val_data = datasets.ImageFolder(self._val_data_dir, self._transform_func)
        return torch.utils.data.DataLoader(dataset=val_data, batch_size=self._batch_size, shuffle=True)


class TorchDataLoader(Builder):
    """
    The Concrete Builder classes follow the Builder interface and provide
    specific implementations of the building steps. Your program may have
    several variations of Builders, implemented differently.
    """

    def __init__(self, data_name: str, n_classes=None, kind=None) -> None:
        """
        A fresh builder instance should contain a blank product object, which is used in further assembly.
        """
        self._data_name = data_name
        self._n_classes = n_classes if n_classes is not None else 2
        self._kind = 'Full' if kind is not None else 'Segments'
        self._data_loader = self.loader

    @property
    def loader(self):
        if self._data_name == 'QUWI':
            self._data_loader = QUWI()
        if self._data_name == 'KHATT_UNIQUE_PARAGRAPH':
            self._data_loader = KHATT_Unique_ParagraphImages(self._n_classes)
        if self._data_name == 'KHATT_FIXED_PARAGRAPH':
            self._data_loader = KHATT_Fixed_ParagraphImages(self._n_classes)
        if self._data_name == 'KHATT_UNIQUE_LINE':
            self._data_loader = KHATT_Unique_LineImages(self._n_classes)
        if self._data_name == 'KHATT_FIXED_LINE':
            self._data_loader = KHATT_Fixed_LineImages(self._n_classes)
        if self._data_name == 'KHATT_COMBINED_LINE':
            self._data_loader = KHATT_Combined_LineImages(self._n_classes)
        if self._data_name == 'KHATT_COMBINED_PARAGRAPH':
            self._data_loader = KHATT_Combined_ParagraphImages(self._n_classes)
        if self._data_name == 'HHD_GENDER':
            self._data_loader = HHD_Gender(self._kind)
        if self._data_name == 'HHD_AGE':
            self._data_loader = HHD_Age(self._kind, self._n_classes)
        if self._data_name == 'MIXED_LANGUAGES':
            self._data_loader = MixedLanguages()
        return self._data_loader

    @property
    def train(self):
        if 'KHATT' in self._data_name:
            return {
                'age': self._data_loader.train()['age'](),
                'gender': self._data_loader.train()['gender']()
            }
        else:
            return self._data_loader.train()

    @property
    def test(self):
        if 'KHATT' in self._data_name:
            return {
                'age': self._data_loader.test()['age'](),
                'gender': self._data_loader.test()['gender']()
            }
        else:
            return self._data_loader.test()

    @property
    def val(self):
        if 'KHATT' in self._data_name:
            return {
                'age': self._data_loader.val()['age'](),
                'gender': self._data_loader.val()['gender']()
            }
        else:
            return self._data_loader.val()
