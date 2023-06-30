import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

def rootDir():
    """
    Get current project absolute path.
    :return: absolute system path of project.
    :rtype: (String)
    """
    abs_path = os.path.abspath('')
    project_dir = os.path.dirname(abs_path)
    return project_dir


COLUMNS = ['RUN_NAME', 'EPOCH', 'TRAIN_LOSS', 'TRAIN_ACCURACY', 'TEST_LOSS', 'TEST_ACCURACY']
stats_dir = rootDir() + '/Stats'

for dir_name in os.listdir(stats_dir):
    plt_title = ' '.join(dir_name.split('_')[:6]) if 'KHATT' in dir_name else ' '.join(dir_name.split('_')[:5])
    if os.path.isdir(os.path.join(stats_dir, dir_name)):
        for filename in os.listdir(os.path.join(stats_dir, dir_name)):
            df = pd.read_csv(os.path.join(stats_dir, dir_name) + '/logs.txt', delimiter='|', names=COLUMNS, header=None)
            df.drop(['RUN_NAME'], axis=1, inplace=True)

            if filename == 'logs.txt':
                sns.lineplot(data=df, x="EPOCH", y="TRAIN_ACCURACY", label='Train', color='green', linestyle='-')
                sns.lineplot(data=df, x="EPOCH", y="TEST_ACCURACY", label='Test', color='blue', linestyle='-')
                plt.title(plt_title)
                plt.legend(loc='lower right')
                plt.xlabel('Epochs')
                plt.ylabel('Accuracy')
                plt.show()
                plt.savefig(stats_dir + f'/{dir_name}/graph.png')
                plt.clf()
