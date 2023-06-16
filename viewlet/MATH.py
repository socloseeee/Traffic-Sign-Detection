import numpy as np
from libs import utils
from numba import njit, prange


@njit(fastmath=True)
def flux(x: float, SCALE: int) -> float:
    return ((2 * (1 / (np.pi ** (1 / 4)))) / (np.sqrt(SCALE))) * (1 - x ** 2) * (1 / (2.71 ** ((x ** 2) / 2)))


@njit(fastmath=True)
def flux_first(x: float, SCALE: int) -> float:
    return (2 * np.sqrt(3) * x * (1 / 2.71 ** ((x ** 2) / 2)) * ((x ** 2) - 3)) / (SCALE * np.pi ** (1 / 4))


@njit(fastmath=True, parallel=True)
def dx_math(pixels: np.ndarray, SCALE: int):
    x_quality, y_quality = pixels.shape
    decomposition = np.int(np.round((np.log(x_quality) / np.log(2)) - 1))
    dwt = np.zeros((y_quality - 1, decomposition, x_quality - 1))
    print("dx_math waiting...")
    for y in prange(y_quality - 1):
        dwt_tmp = np.zeros((decomposition, x_quality - 1))
        for m in prange(decomposition):
            for n in prange(x_quality - 1):
                for x in prange(x_quality - 1):
                    dwt_tmp[m][n] += utils.discrete(x, 2 ** (m - 1), n, flux, SCALE) * pixels[x][y]
        dwt[y] = np.copy(dwt_tmp)
    i_dwt = np.zeros((x_quality - 1, y_quality - 1))
    for y in prange(y_quality - 1):
        for x in prange(x_quality - 1):
            for i in prange(decomposition):
                for j in prange(x_quality - 1):
                    i_dwt[x][y] += utils.discrete(x, 2 ** (i - 1), j, flux_first, SCALE) * dwt[y][i][j]
            i_dwt[x][y] = abs(i_dwt[x][y])
    return i_dwt / 6


@njit(fastmath=True, parallel=True)
def dy_math(pixels: np.ndarray, SCALE):
    x_quality, y_quality = pixels.shape
    decomposition = np.int(np.round((np.log(y_quality) / np.log(2)) - 1))
    dwt = np.zeros((x_quality - 1, decomposition, y_quality - 1))
    print("dy_math waiting...")
    for y in prange(y_quality - 1):
        dwt_tmp = np.zeros((decomposition, y_quality - 1))
        for m in prange(decomposition):
            for n in prange(y_quality - 1):
                for x in prange(y_quality - 1):
                    dwt_tmp[m][n] += utils.discrete(x, 2 ** (m - 1), n, flux, SCALE) * pixels[x][y]
        dwt[y] = np.copy(dwt_tmp)
    i_dwt = np.zeros((y_quality - 1, x_quality - 1))
    for x in prange(x_quality - 1):
        for y in prange(y_quality - 1):
            for i in prange(decomposition):
                for j in prange(y_quality - 1):
                    i_dwt[x][y] += utils.discrete(x, 2 ** (i - 1), j, flux_first, SCALE) * dwt[y][i][j]
            i_dwt[x][y] = abs(i_dwt[x][y])
    return i_dwt / 6
