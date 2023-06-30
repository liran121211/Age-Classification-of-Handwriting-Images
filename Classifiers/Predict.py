import os
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision.models import *
from efficientnet_pytorch import EfficientNet as en
import timm

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


def predict(model, image_path):
    # Define the transformation to be applied to input image
    transform = transforms.Compose([
        transforms.Resize((1000, 1000)),
        transforms.Grayscale(num_output_channels=3),
        transforms.Lambda(to_negative),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Load the input image and apply the defined transformation
    image = Image.open(image_path)
    image = transform(image)

    # Add a batch dimension to input image tensor
    image = image.unsqueeze(0)

    # Set the model to evaluation mode
    model.eval()

    # Pass the input image through the model to get predicted logits
    with torch.no_grad():
        logits = model(image)

    # Apply softmax activation to logits to get predicted class probabilities
    probabilities = torch.softmax(logits, dim=1)

    # Get the predicted class label index
    class_idx = torch.argmax(probabilities, dim=1).item()

    # Return the predicted class label
    return class_idx


def run(folder_path: str, model_name: str, images_folders: list, n_classes: int, classifier_name: str) -> None:
    chosen_dict = None
    two_classes_dict = {0: 'G1', 1: 'G2'}
    three_classes_dict = {0: 'G1', 1: 'G2', 2: 'G3'}
    four_classes_dict = {0: 'G1', 1: 'G2', 2: 'G3', 3: 'G4'}

    if n_classes == 2:
        chosen_dict = two_classes_dict
    if n_classes == 3:
        chosen_dict = three_classes_dict
    if n_classes == 4:
        chosen_dict = four_classes_dict
    if n_classes != 2 and n_classes != 3 and n_classes != 4:
        raise Exception('Number of classes are invalid!\n')

    model_path = fr'{folder_path}/{model_name}'

    # # Model
    # model = en.from_pretrained("efficientnet-b0")
    # model._fc = nn.Linear(model._fc.in_features, n_classes)
    module_dict = torch.load(model_path)['model_state_dict']


    def customModel(n_classes):
        def Efficientnet_B7_():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = en.from_pretrained("efficientnet-b7")
            in_features = model._fc.in_features
            model._fc = nn.Linear(in_features, n_classes)

            model = torch.nn.DataParallel(model)
            model.to(device)
            return model

        def Efficientnet_B0_():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = en.from_pretrained("efficientnet-b0")
            in_features = model._fc.in_features
            model._fc = nn.Linear(in_features, n_classes)

            model = torch.nn.DataParallel(model)
            model.to(device)
            return model

        def NasNet_():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = mnasnet0_5(weights=MNASNet0_5_Weights.DEFAULT)
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, n_classes)

            model = torch.nn.DataParallel(model)
            model.to(device)
            return model

        def VGG16_():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = vgg16(weights=VGG16_Weights.DEFAULT)
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, n_classes)

            model = torch.nn.DataParallel(model)
            model.to(device)
            return model

        def VGG19_():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = vgg19(weights=VGG19_Weights.DEFAULT)
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, n_classes)

            model = torch.nn.DataParallel(model)
            model.to(device)
            return model

        def Inception_V3_():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, n_classes)

            model = torch.nn.DataParallel(model)
            model.to(device)
            return model

        def DenseNet121_():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = densenet121(weights=DenseNet121_Weights.DEFAULT)
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, n_classes)

            model = torch.nn.DataParallel(model)
            model.to(device)
            return model

        def DenseNet169_():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = densenet169(weights=DenseNet169_Weights.DEFAULT)
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, n_classes)

            model = torch.nn.DataParallel(model)
            model.to(device)
            return model

        def EfficientNet_B0_NS():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = timm.create_model('tf_efficientnet_b0_ns', pretrained=True)
            in_features = model.classifier.in_features
            model.classifier= nn.Linear(in_features, n_classes)

            model = torch.nn.DataParallel(model)
            model.to(device)
            return model

        def IG_ResNext101_32x8D():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = timm.create_model('ig_resnext101_32x8d', pretrained=True)
            in_features = model.fc.in_features
            model.fc= nn.Linear(in_features, n_classes)

            model = torch.nn.DataParallel(model)
            model.to(device)
            return model

        def Xception():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = timm.create_model('xception', pretrained=True)
            in_features = model.fc.in_features
            model.fc= nn.Linear(in_features, n_classes)

            model = torch.nn.DataParallel(model)
            model.to(device)
            return model

        def MobileNetv2_100():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = timm.create_model('mobilenetv2_100', pretrained=True)
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, n_classes)

            model = torch.nn.DataParallel(model)
            model.to(device)
            return model

        def ResNetV2_101X1_BITM():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = timm.create_model('resnetv2_50x1_bitm', pretrained=True)
            model.head.fc = nn.Sequential(nn.Linear(1000, 4))

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
            'ResNetV2_101X1_BITM': ResNetV2_101X1_BITM()
        }



    model = customModel(n_classes)[classifier_name]


    # # Fix module dictionary
    # checkpoint = {}
    # for key, val in module_dict.items():
    #     checkpoint[key.replace('module.', '')] = val

    # Load saved model
    model.load_state_dict(torch.load(model_path)['model_state_dict'])

    # Results.txt header
    with open(folder_path + r'/results.txt', 'w+') as f:
        f.write('Prediction|Actual|Result\n')

    # Start prediction
    with open(folder_path + r'/results.txt', 'a') as f:
        # iterate over the files in the folder
        for images_folder in images_folders:
            n_images = len([f for f in os.listdir(images_folder) if os.path.isfile(os.path.join(images_folder, f))])

            for filename in tqdm(os.listdir(images_folder), position=0, desc="Iterations", leave=False, colour='GREEN',
                                 ncols=80, total=n_images):
                # construct the full file path
                file_path = os.path.join(images_folder, filename)
                # check if the file is a file (not a directory) and print its name
                if os.path.isfile(file_path):
                    result = chosen_dict[predict(model, os.path.join(images_folder, filename))]
                    true_label = images_folder.split('/')[-1]
                    f.write(f'{result}|{true_label}|{"SUCCESS" if result == true_label else "FAILED"}')
                    f.write('\n')

#
# # Test 4 Classes
# run(folder_path=rootDir() + r'/Stats/KHATT_Combined_Paragraph_Age_4_Classes_Efficientnet_B0_2023-03-30_03-46-40',
#     model_name='Efficientnet_B0-69.33-2023-03-30_03-46-40.pt',
#     images_folders=[
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/FixedTextParagraphs/Test/Age/4_Classes/G1',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/FixedTextParagraphs/Test/Age/4_Classes/G2',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/FixedTextParagraphs/Test/Age/4_Classes/G3',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/FixedTextParagraphs/Test/Age/4_Classes/G4',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/UniqueTextParagraphs/Test/Age/4_Classes/G1',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/UniqueTextParagraphs/Test/Age/4_Classes/G2',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/UniqueTextParagraphs/Test/Age/4_Classes/G3',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/UniqueTextParagraphs/Test/Age/4_Classes/G4'
#     ],
#     n_classes=4)
#
# # Test 3 Classes
# run(folder_path=rootDir() + r'/Stats/KHATT_Combined_Paragraph_Age_3_Classes_Efficientnet_B0_2023-03-29_17-47-16',
#     model_name='Efficientnet_B0-72.64-2023-03-29_17-47-16.pt',
#     images_folders=[
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/FixedTextParagraphs/Test/Age/3_Classes/G1',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/FixedTextParagraphs/Test/Age/3_Classes/G2',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/FixedTextParagraphs/Test/Age/3_Classes/G3',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/UniqueTextParagraphs/Test/Age/3_Classes/G1',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/UniqueTextParagraphs/Test/Age/3_Classes/G2',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/UniqueTextParagraphs/Test/Age/3_Classes/G3',
#     ],
#     n_classes=3)
#
# # Test 2 Classes
# run(folder_path=rootDir() + r'/Stats/KHATT_Combined_Paragraph_Age_2_Classes_Efficientnet_B0_2023-03-29_17-46-37',
#     model_name='Efficientnet_B0-83.72-2023-03-29_17-46-37.pt',
#     images_folders=[
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/FixedTextParagraphs/Test/Age/2_Classes/G2',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/FixedTextParagraphs/Test/Age/2_Classes/G3',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/UniqueTextParagraphs/Test/Age/2_Classes/G2',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/UniqueTextParagraphs/Test/Age/2_Classes/G3',
#     ],
#     n_classes=2)

# --------------------------------------------------------------------------------------------------------------------
#
# # Test 4 Classes
# run(folder_path=rootDir() + r'/Stats/HHD_Paragraph_Age_4_Classes_Efficientnet_B0_2023-03-30_10-40-37',
#     model_name='Efficientnet_B0-58.62-2023-03-30_10-40-37.pt',
#     images_folders=[
#         rootDir() + r'/Datasets/HHD_Age/Test/Full_Size/4_Classes/G1',
#         rootDir() + r'/Datasets/HHD_Age/Test/Full_Size/4_Classes/G2',
#         rootDir() + r'/Datasets/HHD_Age/Test/Full_Size/4_Classes/G3',
#         rootDir() + r'/Datasets/HHD_Age/Test/Full_Size/4_Classes/G4',
#     ],
#     n_classes=4)
#
# # Test 3 Classes
# run(folder_path=rootDir() + r'/Stats/HHD_Paragraph_Age_3_Classes_Efficientnet_B0_2023-03-30_13-19-59',
#     model_name='Efficientnet_B0-64.35-2023-03-30_13-19-59.pt',
#     images_folders=[
#         rootDir() + r'/Datasets/HHD_Age/Test/Full_Size/3_Classes/G1',
#         rootDir() + r'/Datasets/HHD_Age/Test/Full_Size/3_Classes/G2',
#         rootDir() + r'/Datasets/HHD_Age/Test/Full_Size/3_Classes/G3',
#     ],
#     n_classes=3)
#
# # Test 2 Classes
# run(folder_path=rootDir() + r'/Stats/HHD_Paragraph_Age_2_Classes_Efficientnet_B0_2023-03-30_13-21-43',
#     model_name='Efficientnet_B0-66.36-2023-03-30_13-21-43.pt',
#     images_folders=[
#         rootDir() + r'/Datasets/HHD_Age/Test/Full_Size/2_Classes/G1',
#         rootDir() + r'/Datasets/HHD_Age/Test/Full_Size/2_Classes/G2',
#     ],
#     n_classes=2)
#
# # --------------------------------------------------------------------------------------------------------------------
#
# # Test 4 Classes
# run(folder_path=rootDir() + r'/Stats/KHATT_Fixed_Paragraph_Age_4_Classes_Efficientnet_B0_2023-03-18_09-31-12',
#     model_name='Efficientnet_B0-66.67-2023-03-18_09-31-12.pt',
#     images_folders=[
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/FixedTextParagraphs/Test/Age/4_Classes/G1',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/FixedTextParagraphs/Test/Age/4_Classes/G2',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/FixedTextParagraphs/Test/Age/4_Classes/G3',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/FixedTextParagraphs/Test/Age/4_Classes/G4',
#     ],
#     n_classes=4)
#
# # Test 3 Classes
# run(folder_path=rootDir() + r'/Stats/KHATT_Fixed_Paragraph_Age_3_Classes_Efficientnet_B0_2023-03-18_17-20-19',
#     model_name='Efficientnet_B0-69.39-2023-03-18_17-20-19.pt',
#     images_folders=[
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/FixedTextParagraphs/Test/Age/3_Classes/G1',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/FixedTextParagraphs/Test/Age/3_Classes/G2',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/FixedTextParagraphs/Test/Age/3_Classes/G3',
#     ],
#     n_classes=3)
#
# # Test 2 Classes
# run(folder_path=rootDir() + r'/Stats/KHATT_Fixed_Paragraph_Age_2_Classes_Efficientnet_B0_2023-03-18_09-34-19',
#     model_name='Efficientnet_B0-81.35-2023-03-18_09-34-19.pt',
#     images_folders=[
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/FixedTextParagraphs/Test/Age/2_Classes/G2',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/FixedTextParagraphs/Test/Age/2_Classes/G3',
#     ],
#     n_classes=2)

# --------------------------------------------------------------------------------------------------------------------
#
# # Test 4 Classes
# run(folder_path=rootDir() + r'/Stats/KHATT_Unique_Paragraph_Age_4_Classes_Efficientnet_B0_2023-03-18_20-51-15',
#     model_name='Efficientnet_B0-64.67-2023-03-18_20-51-15.pt',
#     images_folders=[
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/UniqueTextParagraphs/Test/Age/4_Classes/G1',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/UniqueTextParagraphs/Test/Age/4_Classes/G2',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/UniqueTextParagraphs/Test/Age/4_Classes/G3',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/UniqueTextParagraphs/Test/Age/4_Classes/G4',
#     ],
#     n_classes=4)
#
# # Test 3 Classes
# run(folder_path=rootDir() + r'/Stats/KHATT_Unique_Paragraph_Age_3_Classes_Efficientnet_B0_2023-04-01_07-59-37',
#     model_name='Efficientnet_B0-66.67-2023-04-01_07-59-37.pt',
#     images_folders=[
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/UniqueTextParagraphs/Test/Age/3_Classes/G1',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/UniqueTextParagraphs/Test/Age/3_Classes/G2',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/UniqueTextParagraphs/Test/Age/3_Classes/G3',
#     ],
#     n_classes=3)
#
# # Test 2 Classes
# run(folder_path=rootDir() + r'/Stats/KHATT_Unique_Paragraph_Age_2_Classes_Efficientnet_B0_2023-04-01_08-00-09',
#     model_name='Efficientnet_B0-81.35-2023-04-01_08-00-09.pt',
#     images_folders=[
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/UniqueTextParagraphs/Test/Age/2_Classes/G2',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/UniqueTextParagraphs/Test/Age/2_Classes/G3',
#     ],
#     n_classes=2)

# --------------------------------------------------------------------------------------------------------------------


# # Test 4 Classes
# run(folder_path=rootDir() + r'/Stats/KHATT_Combined_Paragraph_Age_4_Classes_Xception_2023-04-02_03-49-58',
#     model_name='Xception-67.83-2023-04-02_03-49-58.pt',
#     images_folders=[
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/FixedTextParagraphs/Test/Age/4_Classes/G1',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/FixedTextParagraphs/Test/Age/4_Classes/G2',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/FixedTextParagraphs/Test/Age/4_Classes/G3',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/FixedTextParagraphs/Test/Age/4_Classes/G4',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/UniqueTextParagraphs/Test/Age/4_Classes/G1',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/UniqueTextParagraphs/Test/Age/4_Classes/G2',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/UniqueTextParagraphs/Test/Age/4_Classes/G3',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/UniqueTextParagraphs/Test/Age/4_Classes/G4'
#     ],
#     n_classes=4, classifier_name='Xception')
#
# # Test 3 Classes
# run(folder_path=rootDir() + r'/Stats/KHATT_Combined_Paragraph_Age_4_Classes_MobileNetv2_100_2023-04-02_03-27-29',
#     model_name='MobileNetv2_100-72.0-2023-04-02_03-27-29.pt',
#     images_folders=[
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/FixedTextParagraphs/Test/Age/3_Classes/G1',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/FixedTextParagraphs/Test/Age/3_Classes/G2',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/FixedTextParagraphs/Test/Age/3_Classes/G3',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/FixedTextParagraphs/Test/Age/4_Classes/G4',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/UniqueTextParagraphs/Test/Age/3_Classes/G1',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/UniqueTextParagraphs/Test/Age/3_Classes/G2',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/UniqueTextParagraphs/Test/Age/3_Classes/G3',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/UniqueTextParagraphs/Test/Age/4_Classes/G4'
#     ],
#     n_classes=4, classifier_name='MobileNetv2_100')
#
# # Test 2 Classes
# run(folder_path=rootDir() + r'/Stats/KHATT_Combined_Paragraph_Age_4_Classes_EfficientNet_B0_NS_2023-04-01_18-42-45',
#     model_name='EfficientNet_B0_NS-69.5-2023-04-01_18-42-45.pt',
#     images_folders=[
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/FixedTextParagraphs/Test/Age/3_Classes/G1',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/FixedTextParagraphs/Test/Age/2_Classes/G2',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/FixedTextParagraphs/Test/Age/2_Classes/G3',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/FixedTextParagraphs/Test/Age/4_Classes/G4',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/UniqueTextParagraphs/Test/Age/3_Classes/G1',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/UniqueTextParagraphs/Test/Age/2_Classes/G2',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/UniqueTextParagraphs/Test/Age/2_Classes/G3',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/UniqueTextParagraphs/Test/Age/4_Classes/G4'
#     ],
#     n_classes=4, classifier_name='EfficientNet_B0_NS')
#
#
# # Test 2 Classes
# run(folder_path=rootDir() + r'/Stats/KHATT_Combined_Paragraph_Age_3_Classes_Xception_2023-04-03_17-05-02',
#     model_name='Xception-71.62-2023-04-03_17-05-02.pt',
#     images_folders=[
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/FixedTextParagraphs/Test/Age/3_Classes/G1',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/FixedTextParagraphs/Test/Age/2_Classes/G2',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/FixedTextParagraphs/Test/Age/2_Classes/G3',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/UniqueTextParagraphs/Test/Age/3_Classes/G1',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/UniqueTextParagraphs/Test/Age/2_Classes/G2',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/UniqueTextParagraphs/Test/Age/2_Classes/G3',
#     ],
#     n_classes=3, classifier_name='Xception')
#
#
# # Test 2 Classes
# run(folder_path=rootDir() + r'/Stats/KHATT_Combined_Paragraph_Age_3_Classes_MobileNetv2_100_2023-04-03_03-32-03',
#     model_name='MobileNetv2_100-71.96-2023-04-03_03-32-03.pt',
#     images_folders=[
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/FixedTextParagraphs/Test/Age/3_Classes/G1',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/FixedTextParagraphs/Test/Age/2_Classes/G2',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/FixedTextParagraphs/Test/Age/2_Classes/G3',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/UniqueTextParagraphs/Test/Age/3_Classes/G1',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/UniqueTextParagraphs/Test/Age/2_Classes/G2',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/UniqueTextParagraphs/Test/Age/2_Classes/G3',
#     ],
#     n_classes=3, classifier_name='MobileNetv2_100')
#
#
# # Test 2 Classes
# run(folder_path=rootDir() + r'/Stats/KHATT_Combined_Paragraph_Age_3_Classes_EfficientNet_B0_NS_2023-04-04_04-02-03',
#     model_name='EfficientNet_B0_NS-73.48-2023-04-04_04-02-03.pt',
#     images_folders=[
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/FixedTextParagraphs/Test/Age/3_Classes/G1',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/FixedTextParagraphs/Test/Age/2_Classes/G2',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/FixedTextParagraphs/Test/Age/2_Classes/G3',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/UniqueTextParagraphs/Test/Age/3_Classes/G1',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/UniqueTextParagraphs/Test/Age/2_Classes/G2',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/UniqueTextParagraphs/Test/Age/2_Classes/G3',
#     ],
#     n_classes=3, classifier_name='EfficientNet_B0_NS')
#
# # Test 2 Classes
# run(folder_path=rootDir() + r'/Stats/KHATT_Combined_Paragraph_Age_2_Classes_Xception_2023-04-03_03-44-38',
#     model_name='Xception-82.75-2023-04-03_03-44-38.pt',
#     images_folders=[
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/FixedTextParagraphs/Test/Age/2_Classes/G2',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/FixedTextParagraphs/Test/Age/2_Classes/G3',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/UniqueTextParagraphs/Test/Age/2_Classes/G2',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/UniqueTextParagraphs/Test/Age/2_Classes/G3',
#     ],
#     n_classes=2, classifier_name='Xception')
#
#
# # Test 2 Classes
# run(folder_path=rootDir() + r'/Stats/KHATT_Combined_Paragraph_Age_2_Classes_MobileNetv2_100_2023-04-02_13-30-46',
#     model_name='MobileNetv2_100-79.84-2023-04-02_13-30-46.pt',
#     images_folders=[
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/FixedTextParagraphs/Test/Age/2_Classes/G2',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/FixedTextParagraphs/Test/Age/2_Classes/G3',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/UniqueTextParagraphs/Test/Age/2_Classes/G2',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/UniqueTextParagraphs/Test/Age/2_Classes/G3',
#     ],
#     n_classes=2, classifier_name='MobileNetv2_100')
#
# # Test 2 Classes
# run(folder_path=rootDir() + r'/Stats/KHATT_Combined_Paragraph_Age_2_Classes_EfficientNet_B0_NS_2023-04-04_04-02-39',
#     model_name='EfficientNet_B0_NS-82.75-2023-04-04_04-02-39.pt',
#     images_folders=[
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/FixedTextParagraphs/Test/Age/2_Classes/G2',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/FixedTextParagraphs/Test/Age/2_Classes/G3',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/UniqueTextParagraphs/Test/Age/2_Classes/G2',
#         rootDir() + r'/Datasets/KHATT/ParagraphImages_v1.0/UniqueTextParagraphs/Test/Age/2_Classes/G3',
#     ],
#     n_classes=2, classifier_name='EfficientNet_B0_NS')

# --------------------------------------------------------------------------------------------------------------------

# run(folder_path=rootDir() + r'/Stats/HHD_Paragraph_Age_4_Classes_Xception_2023-04-05_14-33-18',
#     model_name='Xception-53.45-2023-04-05_14-33-18.pt',
#     images_folders=[
#         rootDir() + r'/Datasets/HHD_Age/Test/Full_Size/4_Classes/G1',
#         rootDir() + r'/Datasets/HHD_Age/Test/Full_Size/4_Classes/G2',
#         rootDir() + r'/Datasets/HHD_Age/Test/Full_Size/4_Classes/G3',
#         rootDir() + r'/Datasets/HHD_Age/Test/Full_Size/4_Classes/G4',
#     ],
#     n_classes=4, classifier_name='Xception')


# run(folder_path=rootDir() + r'/Stats/HHD_Paragraph_Age_4_Classes_MobileNetv2_100_2023-04-05_14-36-28',
#     model_name='MobileNetv2_100-65.52-2023-04-05_14-36-28.pt',
#     images_folders=[
#         rootDir() + r'/Datasets/HHD_Age/Test/Full_Size/4_Classes/G1',
#         rootDir() + r'/Datasets/HHD_Age/Test/Full_Size/4_Classes/G2',
#         rootDir() + r'/Datasets/HHD_Age/Test/Full_Size/4_Classes/G3',
#         rootDir() + r'/Datasets/HHD_Age/Test/Full_Size/4_Classes/G4',
#     ],
#     n_classes=4, classifier_name='MobileNetv2_100')


# run(folder_path=rootDir() + r'/Stats/HHD_Paragraph_Age_4_Classes_EfficientNet_B0_NS_2023-04-04_15-52-54',
#     model_name='EfficientNet_B0_NS-53.45-2023-04-04_15-52-54.pt',
#     images_folders=[
#         rootDir() + r'/Datasets/HHD_Age/Test/Full_Size/4_Classes/G1',
#         rootDir() + r'/Datasets/HHD_Age/Test/Full_Size/4_Classes/G2',
#         rootDir() + r'/Datasets/HHD_Age/Test/Full_Size/4_Classes/G3',
#         rootDir() + r'/Datasets/HHD_Age/Test/Full_Size/4_Classes/G4',
#     ],
#     n_classes=4, classifier_name='EfficientNet_B0_NS')

# run(folder_path=rootDir() + r'/Stats/HHD_Paragraph_Age_4_Classes_Efficientnet_B0_2023-03-30_10-40-37',
#     model_name='Efficientnet_B0-58.62-2023-03-30_10-40-37.pt',
#     images_folders=[
#         rootDir() + r'/Datasets/HHD_Age/Test/Full_Size/4_Classes/G1',
#         rootDir() + r'/Datasets/HHD_Age/Test/Full_Size/4_Classes/G2',
#         rootDir() + r'/Datasets/HHD_Age/Test/Full_Size/4_Classes/G3',
#         rootDir() + r'/Datasets/HHD_Age/Test/Full_Size/4_Classes/G4',
#     ],
#     n_classes=4, classifier_name='Efficientnet_B0')

# run(folder_path=rootDir() + r'/Stats/HHD_Paragraph_Age_3_Classes_Xception_2023-04-05_05-31-24',
#     model_name='Xception-53.04-2023-04-05_05-31-24.pt',
#     images_folders=[
#         rootDir() + r'/Datasets/HHD_Age/Test/Full_Size/3_Classes/G1',
#         rootDir() + r'/Datasets/HHD_Age/Test/Full_Size/3_Classes/G2',
#         rootDir() + r'/Datasets/HHD_Age/Test/Full_Size/3_Classes/G3',
#     ],
#     n_classes=3, classifier_name='Xception')
#
# run(folder_path=rootDir() + r'/Stats/HHD_Paragraph_Age_3_Classes_MobileNetv2_100_2023-04-05_14-35-47',
#     model_name='MobileNetv2_100-66.09-2023-04-05_14-35-47.pt',
#     images_folders=[
#         rootDir() + r'/Datasets/HHD_Age/Test/Full_Size/3_Classes/G1',
#         rootDir() + r'/Datasets/HHD_Age/Test/Full_Size/3_Classes/G2',
#         rootDir() + r'/Datasets/HHD_Age/Test/Full_Size/3_Classes/G3',
#     ],
#     n_classes=3, classifier_name='MobileNetv2_100')
#
# run(folder_path=rootDir() + r'/Stats/HHD_Paragraph_Age_3_Classes_EfficientNet_B0_NS_2023-04-04_15-52-22',
#     model_name='EfficientNet_B0_NS-56.52-2023-04-04_15-52-22.pt',
#     images_folders=[
#         rootDir() + r'/Datasets/HHD_Age/Test/Full_Size/3_Classes/G1',
#         rootDir() + r'/Datasets/HHD_Age/Test/Full_Size/3_Classes/G2',
#         rootDir() + r'/Datasets/HHD_Age/Test/Full_Size/3_Classes/G3',
#     ],
#     n_classes=3, classifier_name='EfficientNet_B0_NS')
#
# run(folder_path=rootDir() + r'/Stats/HHD_Paragraph_Age_3_Classes_Efficientnet_B0_2023-03-30_13-19-59',
#     model_name='Efficientnet_B0-64.35-2023-03-30_13-19-59.pt',
#     images_folders=[
#         rootDir() + r'/Datasets/HHD_Age/Test/Full_Size/3_Classes/G1',
#         rootDir() + r'/Datasets/HHD_Age/Test/Full_Size/3_Classes/G2',
#         rootDir() + r'/Datasets/HHD_Age/Test/Full_Size/3_Classes/G3',
#     ],
#     n_classes=3, classifier_name='Efficientnet_B0')

run(folder_path=rootDir() + r'/Stats/HHD_Paragraph_Age_2_Classes_Xception_2023-04-05_05-28-46',
    model_name='Xception-65.45-2023-04-05_05-28-46.pt',
    images_folders=[
        rootDir() + r'/Datasets/HHD_Age/Test/Full_Size/2_Classes/G1',
        rootDir() + r'/Datasets/HHD_Age/Test/Full_Size/2_Classes/G2',
    ],
    n_classes=2, classifier_name='Xception')

run(folder_path=rootDir() + r'/Stats/HHD_Paragraph_Age_2_Classes_EfficientNet_B0_NS_2023-04-05_05-24-10',
    model_name='EfficientNet_B0_NS-57.27-2023-04-05_05-24-10.pt',
    images_folders=[
        rootDir() + r'/Datasets/HHD_Age/Test/Full_Size/2_Classes/G1',
        rootDir() + r'/Datasets/HHD_Age/Test/Full_Size/2_Classes/G2',
    ],
    n_classes=2, classifier_name='EfficientNet_B0_NS')

run(folder_path=rootDir() + r'/Stats/HHD_Paragraph_Age_2_Classes_Efficientnet_B0_2023-03-30_13-21-43',
    model_name='Efficientnet_B0-66.36-2023-03-30_13-21-43.pt',
    images_folders=[
        rootDir() + r'/Datasets/HHD_Age/Test/Full_Size/2_Classes/G1',
        rootDir() + r'/Datasets/HHD_Age/Test/Full_Size/2_Classes/G2',
    ],
    n_classes=2, classifier_name='Efficientnet_B0')

run(folder_path=rootDir() + r'/Stats/HHD_Paragraph_Age_2_Classes_EfficientNet_B0_NS_2023-04-05_05-24-10',
    model_name='EfficientNet_B0_NS-57.27-2023-04-05_05-24-10.pt',
    images_folders=[
        rootDir() + r'/Datasets/HHD_Age/Test/Full_Size/2_Classes/G1',
        rootDir() + r'/Datasets/HHD_Age/Test/Full_Size/2_Classes/G2',
    ],
    n_classes=2, classifier_name='EfficientNet_B0_NS')

run(folder_path=rootDir() + r'/Stats/HHD_Paragraph_Age_2_Classes_MobileNetv2_100_2023-04-06_08-49-20',
    model_name='MobileNetv2_100-68.18-2023-04-06_08-49-20.pt',
    images_folders=[
        rootDir() + r'/Datasets/HHD_Age/Test/Full_Size/2_Classes/G1',
        rootDir() + r'/Datasets/HHD_Age/Test/Full_Size/2_Classes/G2',
    ],
    n_classes=2, classifier_name='MobileNetv2_100')