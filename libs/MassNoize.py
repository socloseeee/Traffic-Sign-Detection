import numpy as np
from PIL import Image


def gaussian_noise(image, width, height, scale):
    result = Image.new('RGB', (width, height))
    for x in np.arange(width):
        for y in np.arange(height):
            pixel = np.int_(image.getpixel((x, y))[0] + np.random.normal(0, scale))
            result.putpixel((x, y), (pixel, pixel, pixel))
    return result
