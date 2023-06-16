from PIL import Image
import numpy as np
from tqdm import tqdm


def sob_matrix(image, width, height):
    m_gx = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
    m_gy = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
    result = Image.new('RGB', (width, height))
    with tqdm(np.arange(1, width - 2, 1), ncols=100, desc="grad") as t:
        for iX in t:
            for iY in np.arange(1, height - 2, 1):
                gx, gy = 0, 0
                tmp_result = sub_matrix(image, iX-1, iX+1, iY-1, iY+1)
                for i in np.arange(3):
                    for j in np.arange(3):
                        gx += tmp_result[i][j] * m_gx[i][j]
                        gy += tmp_result[i][j] * m_gy[i][j]
                pixel = np.int_(np.sqrt((np.square(gx) + np.square(gy))))
                result.putpixel((iX, iY), (pixel, pixel, pixel))
    return result


def sub_matrix(image, startRow, lastRow, startCol, lastCol):
    sub_struct = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    row, col = 0, 0
    for i in np.arange(startRow, lastRow + 1, 1):
        for j in np.arange(startCol, lastCol + 1, 1):
            tmp = np.int_(image.getpixel((i, j))[0])
            sub_struct[row][col] = tmp
            col += 1
        row += 1
        col = 0
    return sub_struct
