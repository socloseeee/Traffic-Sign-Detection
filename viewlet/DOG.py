import math
import numpy as np
from numba import njit, prange


@njit(fastmath=True)
def flux(x: float) -> float:
    return (1 / 2.71 ** (x ** 2 / 2)) - 0.5 * (1 / 2.71 ** (x ** 2 / 8))


@njit(fastmath=True)
def flux_first(x: float) -> float:
    return 0.125 * x * (1 / 2.71 ** (x ** 2 / 8)) - x * (1 / 2.71 ** (x ** 2 / 2))


@njit(fastmath=True)
def discrete(x: float, m: float, n: float, callback, SCALE: int):
    return 1 / 1 / (SCALE ** (m / 2)) * callback((1 / (SCALE ** m)) * x - n)


@njit(fastmath=True, parallel=True)
def dwt_x(pixels: np.ndarray, SCALE: int):
    print("dwt_x waiting...")
    Xq, Yq = pixels.shape
    Xdec = int(round((math.log(Xq) / math.log(2)) - 1))
    tmp = np.zeros((Yq - 1, Xdec, Xq - 1))
    for y in prange(Yq - 1):
        tmp_result = np.zeros((Xdec, Xq - 1))
        for m in prange(Xdec):
            for n in prange(Xq - 1):
                for x in prange(Xq - 1):
                    tmp_result[m][n] += discrete(x, 2 ** (m - 1), n, flux, SCALE) * pixels[x][y]
        tmp[y] = tmp_result
    i_tmp = np.zeros((Xq - 1, Yq - 1))
    for y in prange(Yq - 1):
        for i in prange(Xdec):
            for j in prange(Xq - 1):
                for x in prange(Xq - 1):
                    i_tmp[x][y] += discrete(x, 2 ** (i - 1), j, flux_first, SCALE) * tmp[y][i][j]
        i_tmp[:, y] = np.abs(i_tmp[:, y])
    return i_tmp / 2.0


@njit(fastmath=True, parallel=True)
def dwt_y(pixels: np.ndarray, SCALE: int):
    Xq, Yq = pixels.shape
    Ydec = int(round((math.log(Yq) / math.log(2)) - 1))
    tmp = np.zeros((Xq - 1, Ydec, Yq - 1))
    print("dwt_y waiting...")
    for y in prange(Yq - 1):
        for m in prange(Ydec):
            for n in prange(Yq - 1):
                tmp_result = np.zeros((Yq - 1))
                for x in prange(Yq - 1):
                    tmp_result[n] += discrete(x, 2 ** (m - 1), n, flux, SCALE) * pixels[x][y]
                tmp[y][m][n] = tmp_result[n]
    i_tmp = np.zeros((Yq - 1, Xq - 1))
    for x in prange(Xq - 1):
        for y in prange(Yq - 1):
            for i in prange(Ydec):
                tmp_result = np.zeros((Yq - 1))
                for j in prange(Yq - 1):
                    tmp_result[j] += discrete(x, 2 ** (i - 1), j, flux_first, SCALE) * tmp[y][i][j]
                i_tmp[x][y] += tmp_result[j]
            i_tmp[x][y] = np.abs(i_tmp[x][y])
    return i_tmp / 2.0

