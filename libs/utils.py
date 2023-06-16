import os

import PIL
import cv2
import numpy as np
from PIL import Image

from tqdm import tqdm
from PyQt5 import QtCore
from numba import njit, prange
from PyQt5.QtWidgets import QFileDialog, QDialog


@njit(fastmath=True, parallel=True)
def rs_chm(pixels: np.ndarray, vector: str):
    res = np.zeros(pixels.shape)
    for x in prange(1, pixels.shape[0] - 1):
        for y in prange(1, pixels.shape[1] - 1):
            if vector == 'x':
                res[y][x] = pixels[y][x] - pixels[y][x - 1]
            else:
                res[y][x] = pixels[y][x] - pixels[y - 1][x]
    return res


def grad(image, dif_x, dif_y):
    result = image.copy()
    with tqdm(np.arange(dif_x.shape[0]), ncols=100, desc="grad") as t:
        for x in t:
            for y in np.arange(dif_y.shape[0]):
                pixel = np.int_(np.round(np.sqrt(np.square(dif_x[x][y]) + np.square(dif_y[x][y]))))
                result.putpixel((x, y), (pixel, pixel, pixel))
    return result


@njit(fastmath=True)
def discrete(x: float, m: float, n: float, callback, SCALE: int):
    return 1 / (SCALE ** (m / 2)) * callback((1 / (SCALE ** m)) * x - n, SCALE)


def threshold(image: PIL.Image, width, height):
    resource = image.copy()
    with tqdm(prange(width), ncols=100, desc="threshold", colour="green") as t:
        for i in t:
            for j in prange(height):
                pixel = resource.getpixel((i, j))[0]
                if pixel > 90:
                    resource.putpixel((i, j), (255, 255, 255))
                else:
                    resource.putpixel((i, j), (0, 0, 0))
    return resource


def sko(img1, img2):
    result = 0
    for i in prange(img1.size[0]):
        for j in prange(img1.size[1]):
            result += (img1.getpixel((i, j))[0] - img2.getpixel((i, j))[0]) ** 2
    return np.sqrt(result + ((img1.size[0] - 1) * (img1.size[1])))


def FileDialog(directory='', forOpen=True, fmt='', isFolder=False) -> str:
    options = QFileDialog.Options()
    options |= QFileDialog.DontUseNativeDialog
    options |= QFileDialog.DontUseCustomDirectoryIcons
    dialog = QFileDialog()
    dialog.setOptions(options)

    dialog.setFilter(dialog.filter() | QtCore.QDir.Hidden)

    # ARE WE TALKING ABOUT FILES OR FOLDERS
    if isFolder:
        dialog.setFileMode(QFileDialog.DirectoryOnly)
    else:
        dialog.setFileMode(QFileDialog.AnyFile)
    # OPENING OR SAVING
    dialog.setAcceptMode(QFileDialog.AcceptOpen) if forOpen else dialog.setAcceptMode(QFileDialog.AcceptSave)

    # SET FORMAT, IF SPECIFIED
    if fmt != '' and isFolder is False:
        dialog.setDefaultSuffix(fmt)
        dialog.setNameFilters([f'{fmt} (*.{fmt})'])

    # SET THE STARTING DIRECTORY
    if directory != '':
        dialog.setDirectory(str(directory))
    else:
        dialog.setDirectory(str(os.getcwd()))

    if dialog.exec_() == QDialog.Accepted:
        path = dialog.selectedFiles()[0]  # returns a list
        return path
    else:
        return ''


def ImgCrop(image_path: str, model):
    image = cv2.imread(image_path)
    predict = model.predict(image_path, confidence=10, overlap=30).json()
    try:
        if predict['predictions']:
            img_prediction_dict = predict["predictions"][0]
            center_x, center_y, width, height = list(img_prediction_dict.values())[:4]
            x1, y1, x2, y2 = center_x - (width // 2), center_y - (height // 2), center_x + (width // 2), center_y + (
                    height // 2)
            cropped_img = image[y1:y2, x1:x2]
        else:
            cropped_img = image
    except Exception as e:
        print(e)
        cropped_img = image
    cropped_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
    return cropped_img
