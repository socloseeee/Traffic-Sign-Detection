from PIL import Image
import numpy as np
from tqdm import tqdm


def get_image_info(filename):
    try:
        img = Image.open(filename)
        img_resized = img.resize((512, 512))
        # img_resized = img.resize((512, 512))
        with img_resized as image:
            matrix = image.load()
            width = image.size[0]
            height = image.size[1]
            pixels = np.zeros((width, height), dtype=np.int)
            with tqdm(np.arange(width), ncols=100, desc="image to array", colour="blue") as t:
                for i in t:
                    for j in np.arange(height):
                        pixels[i][j] = matrix[i, j][0]

            return [image, width, height, pixels, matrix]
    except ImportError:
        print('Ошибка открытия файла')


def get_image_info2(img):
    try:
        img_resized = img.resize((512, 512))
        with img_resized as image:
            matrix = image.load()
            width = image.size[0]
            height = image.size[1]
            pixels = np.zeros((width, height), dtype=np.int)
            with tqdm(np.arange(width), ncols=100, desc="image to array", colour="blue") as t:
                for i in t:
                    for j in np.arange(height):
                        pixels[i][j] = matrix[i, j][0]
            return [image, width, height, pixels, matrix]
    except ImportError:
        print('Ошибка открытия файла')

# def decomposition(filed: str) -> dict:
#     result = int((np.log10(get_image_info(filename)[filed]) / np.log10(2)) - 1)
#     return {
#         'm': np.arange(0, result),
#         'n': np.arange(0, result - 1),
#         'k': np.arange(0, result - 1)
#     }
