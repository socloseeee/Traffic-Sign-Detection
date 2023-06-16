import math
import numpy as np
from numba import njit, prange


@njit(fastmath=True)
def flux(x: float) -> float:
    return -x * (1 / 2.71 ** ((x ** 2) / 2))


@njit(fastmath=True)
def flux_first(x: float) -> float:
    return (x ** 2) * (1 / 2.71 ** ((x ** 2) / 2)) - (1 / 2.71 ** ((1 / 2) * x ** 2))


@njit(fastmath=True)
def discrete(x: float, m: float, n: float, callback, SCALE: int) -> float:
    return 1 / (3 ** (m / 2)) * callback((1 / (SCALE ** m)) * x - n)


@njit(fastmath=True, parallel=True)
def dwt_x(pixels: np.ndarray, SCALE: int):
    Xq, Yq = pixels.shape
    Xdec = int(round((math.log(Xq) / math.log(2)) - 1))
    tmp = np.zeros((Yq - 1, Xdec, Xq - 1))
    print("dwt_x waiting...")
    for y in prange(Yq - 1):
        tmp_result = np.zeros((Xdec, Xq - 1))
        for m in prange(Xdec):
            for n in prange(Xq - 1):
                for x in prange(Xq - 1):
                    tmp_result[m][n] += discrete(x, 2 ** (m - 1), n, flux, SCALE) * pixels[x][y]
        tmp[y] = np.copy(tmp_result)
    i_tmp = np.zeros((Xq - 1, Yq - 1))
    for y in prange(Yq - 1):
        for x in prange(Xq - 1):
            for i in prange(Xdec):
                for j in prange(Xq - 1):
                    i_tmp[x][y] += discrete(x, 2 ** (i - 1), j, flux_first, SCALE) * tmp[y][i][j]
            i_tmp[x][y] = abs(i_tmp[x][y])
    return i_tmp / 6


@njit(fastmath=True, parallel=True)
def dwt_y(pixels, SCALE: int):
    Xq, Yq = pixels.shape
    Ydec = int(round((math.log(Yq) / math.log(2)) - 1))
    tmp = np.zeros((Xq - 1, Ydec, Yq - 1))
    print("dwt_y waiting...")
    for y in prange(Yq - 1):
        tmp_result = np.zeros((Ydec, Yq - 1))
        for m in prange(Ydec):
            for n in prange(Yq - 1):
                for x in prange(Yq - 1):
                    tmp_result[m][n] += discrete(x, 2 ** (m - 1), n, flux, SCALE) * pixels[x][y]
        tmp[y] = tmp_result.copy()
    i_tmp = np.zeros((Yq - 1, Xq - 1))
    for x in prange(Xq - 1):
        for y in prange(Yq - 1):
            for i in prange(Ydec):
                for j in prange(Yq - 1):
                    i_tmp[x][y] += discrete(x, 2 ** (i - 1), j, flux_first, SCALE) * tmp[y][i][j]
            i_tmp[x][y] = abs(i_tmp[x][y])
    return i_tmp / 6
