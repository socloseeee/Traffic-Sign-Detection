from numba import prange
import numpy as np
from tqdm import tqdm


def normalize_image(image, matrix, width, height):
    result = image.copy()
    min_c, max_c = matrix[0, 0][0], matrix[0, 0][0]
    with tqdm(prange(width), ncols=100, desc="max_c&min_c") as t:
        for x in t:
            for y in prange(height):
                if min_c > matrix[x, y][0]:
                    min_c = matrix[x, y][0]
                if max_c < matrix[x, y][0]:
                    max_c = matrix[x, y][0]
    with tqdm(prange(width), ncols=100, desc="serialize pixels") as t:
        for x in t:
            for y in prange(height):
                pixel = np.int_(((image.getpixel((x, y))[0] - min_c) * 254) / (max_c - min_c))
                result.putpixel((x, y), (pixel, pixel, pixel))
    return result
