from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import pandas as pd
import os
import sys

sys.path.append(r'/home/unknown/python_files/')


def rootDir():
    """
    Get current project absolute path.
    :return: absolute system path of project.
    :rtype: (String)
    """
    abs_path = os.path.abspath('')
    project_dir = os.path.dirname(abs_path)
    return project_dir


def create_confusion_matrix(name: str, file_path: str, n_classes: int, dataset_type: str) -> None:
    if dataset_type == 'KHATT':
        if n_classes == 2:
            classes = ['G2', 'G3']
        if n_classes == 3:
            classes = ['G1', 'G2', 'G3']
        if n_classes == 4:
            classes = ['G1', 'G2', 'G3', 'G4']

    if dataset_type == 'HHD':
        if n_classes == 2:
            classes = ['G1', 'G2']
        if n_classes == 3:
            classes = ['G1', 'G2', 'G3']
        if n_classes == 4:
            classes = ['G1', 'G2', 'G3', 'G4']

    df = pd.read_csv(file_path, delimiter='|')
    conf_matrix = confusion_matrix(y_true=df['Actual'], y_pred=df['Prediction'])
    _, _ = plot_confusion_matrix(conf_mat=conf_matrix, figsize=(6, 6), cmap=plt.cm.Greens, class_names=classes)
    plt.title(name)
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actual', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.savefig('/'.join(file_path.split('/')[:-1]) + '/' + 'confusion_matrix.png')


stats_dir = rootDir() + '/Stats'

# HHD Stats
# for dir_name in os.listdir(stats_dir):
#     confusion_matrix_title = " ".join(dir_name.split('_')[:4])
#
#     if os.path.isdir(os.path.join(stats_dir, dir_name)):
#         if 'HHD_Paragraph' in dir_name:
#             for filename in os.listdir(os.path.join(stats_dir, dir_name)):
#                 file_full_path = rootDir() + '/Stats/' + dir_name + '/' + filename
#                 n_classes = int("".join(dir_name.split('_')[3]))
#                 if filename == 'results.txt':
#                     create_confusion_matrix(confusion_matrix_title, file_full_path, n_classes, 'HHD')

# KHATT Stats
for dir_name in os.listdir(stats_dir):
    confusion_matrix_title = " ".join(dir_name.split('_')[:5])

    if os.path.isdir(os.path.join(stats_dir, dir_name)):
        if 'KHATT_Combined_Paragraph_Age_4_Classes_RexNet_100_2023-04-06_14-12-39' in dir_name:
            for filename in os.listdir(os.path.join(stats_dir, dir_name)):
                file_full_path = rootDir() + '/Stats/' + dir_name + '/' + filename
                print(file_full_path)
                n_classes = int("".join(dir_name.split('_')[4]))
                if filename == 'results.txt':
                    create_confusion_matrix(confusion_matrix_title, file_full_path, n_classes, 'HHD')