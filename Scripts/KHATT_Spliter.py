import glob
import os
from PIL import Image
import pandas as pd

# GLOBAL VARS
os.chdir('..')
IMAGES_DIR = os.getcwd() + r"/Datasets/KHATT_FIXED/P_Validate"
DESCRIPTOR_PATH = os.getcwd() + r"/Datasets/KHATT/KHATT_Statistics.xlsx"


def load() -> pd.DataFrame:
    """
    load XLSX statistics file about the images.
    @return: pandas DataFrame with (name, age group, gender) columns.
    @rtype: pd.DataFrame
    """
    df = pd.DataFrame(pd.read_excel(DESCRIPTOR_PATH), columns=['Form Number', 'Age Group', 'Gender'])
    return df


def getImageData(df: pd.DataFrame, img_id: str) -> tuple:
    """
    Find image details (name, age group, gender).
    @param df: KHATT statistics file.
    @type df: pandas DataFrame
    @param img_id: image name.
    @type img_id: string
    @return: string values of (name, age group, gender).
    @rtype: tuple
    """
    result = df.loc[df['Form Number'] == img_id]
    if result.empty:
        print(f'{img_id} image could not be found.')
        return 'NA', 'NA', 'NA'

    return str(result.iloc[0, 0]), str(result.iloc[0, 1]), str(result.iloc[0, 2])


def splitToDirs(df: pd.DataFrame) ->None:
    """
    Split images ti directories according to gender and age group.
    @param df: KHATT statistics file.
    @type df: pandas DataFrame
    """
    n_files = len(glob.glob1(IMAGES_DIR, "*.tif"))

    for i, file_name in enumerate(os.listdir(IMAGES_DIR)):
        print(f'\rSplitting {i+1}/{n_files} KHATT images.', end='')

        file_path = os.path.join(IMAGES_DIR, file_name)
        # checking if it is a file
        if os.path.isfile(file_path):
            file_name = file_name[5:-4].split('_')[0]
            img_copy = Image.open(file_path).copy()
            file_descriptor = getImageData(df, file_name)

            # skip non-exists images
            if file_descriptor == ('NA', 'NA', 'NA'):
                continue

            # split age group folders
            if file_descriptor[1] == '1':
                img_copy.save(os.getcwd() + r"/Datasets/KHATT/Paragraph_Images/age/val/group_1/{0}.{1}".format(file_name, 'jpg'))
            elif file_descriptor[1] == '2':
                img_copy.save(os.getcwd() + r"/Datasets/KHATT/Paragraph_Images/age/val/group_2/{0}.{1}".format(file_name, 'jpg'))
            elif file_descriptor[1] == '3':
                img_copy.save(os.getcwd() + r"/Datasets/KHATT/Paragraph_Images/age/val/group_3/{0}.{1}".format(file_name, 'jpg'))
            elif file_descriptor[1] == '4':
                img_copy.save(os.getcwd() + r"/Datasets/KHATT/Paragraph_Images/age/val/group_4/{0}.{1}".format(file_name, 'jpg'))

            # split gender folders
            if file_descriptor[2] == 'M':
                img_copy.save(os.getcwd() + r"/Datasets/KHATT/Paragraph_Images/gender/val/male/{0}.{1}".format(file_name, 'jpg'))
            elif file_descriptor[2] == 'F':
                img_copy.save(os.getcwd() + r"/Datasets/KHATT/Paragraph_Images/gender/val/female/{0}.{1}".format(file_name, 'jpg'))
