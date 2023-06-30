import cv2
import os
import glob
from numpy import ndarray
import sys

# CMD run configuration
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


def countNotWhitePixels(im, black_px_min):
    """count how many not white pixels in given image
     :parameter
     im: image to count black pixels
     black_px_min: threshold of the minimum of black pixels
     :return
     (bool): true if there is more black pixels than the threshold, else false
    """
    black = 0
    for i in range(im.shape[0]):  # traverses through height of the image
        for j in range(im.shape[1]):  # traverses through width of the image
            if im[i][j] < 200:
                black += 1
    return black > black_px_min  # return true if there is more black pixels than minimum black pixels


def imageSegment(image: ndarray) -> list:
    """divide image using 'slide window' to patches ,each patch is 400x400 scale
    :parameter
    image : make as many patches as can from this image
    :returns
    images: list of all patches
    """
    height = len(image)  # y is height
    width = len(image[0])  # x is width
    y = 0
    images = []
    while y <= height - 400:
        x = 100
        while x < width - 200:
            div_im = image[y:y + 400, x:x + 400].copy()
            if countNotWhitePixels(div_im, 5000):
                images.append(div_im)
            x = x + 200
        y = y + 200

    return images


def makeSegmentation(source_path, dest_path, img_w, img_h) -> None:
    """
    make patches from all images with in source path, and save them at destination path
    source_path : directory source path of images (str).
    dest_path : directory destination path of images (str).
    return : None
    """
    n_files = len(glob.glob1(source_path, "*.tif"))
    f = 0
    for filename in os.listdir(source_path):
        f += 1
        print(f'\r{source_path} - Processing - {f} / {n_files}', end='')
        image = cv2.imread(source_path + '/' + filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (img_w, img_h))
        images = imageSegment(image)

        # if destination directory does not exist, create it.
        try:
            os.mkdir(dest_path)
        except OSError:
            print("Creation of the directory %s failed" % dest_path)
        else:
            print("Successfully created the directory %s " % dest_path)

        # save image patches.
        for patch_id in range(len(images)):
            cv2.imwrite(dest_path + '\\' + filename.replace('.tif', '_') + str(patch_id + 1) + '.jpg', images[patch_id])


if __name__ == "__main__":
    SRC_DIR = rootDir() + r"/Datasets/HHD_Age/Train/Full_Size/G1/"
    DEST_DIR = rootDir() + r"/Datasets/HHD_Age/Train/Segments/G1/"
    makeSegmentation(SRC_DIR, DEST_DIR, 1500, 1200)

    SRC_DIR = rootDir() + r"/Datasets/HHD_Age/Train/Full_Size/G2/"
    DEST_DIR = rootDir() + r"/Datasets/HHD_Age/Train/Segments/G2/"
    makeSegmentation(SRC_DIR, DEST_DIR, 1500, 1200)

    SRC_DIR = rootDir() + r"/Datasets/HHD_Age/Train/Full_Size/G3/"
    DEST_DIR = rootDir() + r"/Datasets/HHD_Age/Train/Segments/G3/"
    makeSegmentation(SRC_DIR, DEST_DIR, 1500, 1200)

    SRC_DIR = rootDir() + r"/Datasets/HHD_Age/Train/Full_Size/G4/"
    DEST_DIR = rootDir() + r"/Datasets/HHD_Age/Train/Segments/G4/"
    makeSegmentation(SRC_DIR, DEST_DIR, 1500, 1200)




    SRC_DIR = rootDir() + r"/Datasets/HHD_Age/Validation/Full_Size/G1/"
    DEST_DIR = rootDir() + r"/Datasets/HHD_Age/Validation/Segments/G1/"
    makeSegmentation(SRC_DIR, DEST_DIR, 1500, 1200)

    SRC_DIR = rootDir() + r"/Datasets/HHD_Age/Validation/Full_Size/G2/"
    DEST_DIR = rootDir() + r"/Datasets/HHD_Age/Validation/Segments/G2/"
    makeSegmentation(SRC_DIR, DEST_DIR, 1500, 1200)

    SRC_DIR = rootDir() + r"/Datasets/HHD_Age/Validation/Full_Size/G3/"
    DEST_DIR = rootDir() + r"/Datasets/HHD_Age/Validation/Segments/G3/"
    makeSegmentation(SRC_DIR, DEST_DIR, 1500, 1200)

    SRC_DIR = rootDir() + r"/Datasets/HHD_Age/Validation/Full_Size/G4/"
    DEST_DIR = rootDir() + r"/Datasets/HHD_Age/Validation/Segments/G4/"
    makeSegmentation(SRC_DIR, DEST_DIR, 1500, 1200)



    SRC_DIR = rootDir() + r"/Datasets/HHD_Age/Test/Full_Size/G1/"
    DEST_DIR = rootDir() + r"/Datasets/HHD_Age/Test/Segments/G1/"
    makeSegmentation(SRC_DIR, DEST_DIR, 1500, 1200)

    SRC_DIR = rootDir() + r"/Datasets/HHD_Age/Test/Full_Size/G2/"
    DEST_DIR = rootDir() + r"/Datasets/HHD_Age/Test/Segments/G2/"
    makeSegmentation(SRC_DIR, DEST_DIR, 1500, 1200)

    SRC_DIR = rootDir() + r"/Datasets/HHD_Age/Test/Full_Size/G3/"
    DEST_DIR = rootDir() + r"/Datasets/HHD_Age/Test/Segments/G3/"
    makeSegmentation(SRC_DIR, DEST_DIR, 1500, 1200)

    SRC_DIR = rootDir() + r"/Datasets/HHD_Age/Test/Full_Size/G4/"
    DEST_DIR = rootDir() + r"/Datasets/HHD_Age/Test/Segments/G4/"
    makeSegmentation(SRC_DIR, DEST_DIR, 1500, 1200)