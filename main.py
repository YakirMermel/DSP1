import numpy as np
import math
from PIL import Image
import cmath as cm
import matplotlib.pyplot as plt


def round_to(z, k):
    real = round((10 ** k) * np.real(z)) / (10 ** k)
    imag = round((10 ** k) * np.imag(z)) / (10 ** k)
    return complex(real, imag)


def fft(x):
    length = len(x)
    if not ((math.log(length, 2)).is_integer()):
        return []

    if length == 1:
        return x
    else:
        coe_even = fft(x[::2])
        coe_odd = fft(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(length) / length)

        coe = np.concatenate(
            [coe_even + factor[:int(length / 2)] * coe_odd, coe_even + factor[int(length / 2):] * coe_odd])
        return [round_to(num, 10) for num in coe]


def ifft(x):
    return [num / len(x) for num in np.roll(np.flip(fft(x)), 1)]


def dtft(x, w):  # calculate X(e^(jwn)) for certain w
    result = 0
    for n in range(len(x)):
        result += x[n] * cm.exp(-1j * w * n)
    return result


def delta(n):
    if n == 0:
        return 1
    else:
        return 0


def f1(n, m):
    if n <= 7 and m <= 7:
        return 1
    else:
        return 0


def dft2d(mat):
    m = np.array([np.fft.fft(mat[i]) for i in range(len(mat))]).transpose()
    return np.array([np.fft.fft(m[i]) for i in range(len(m))]).transpose()


def cyclic_conv(arr1, arr2):
    if len(arr1) >= len(arr2):
        arr2 += [0] * (len(arr1) - len(arr2))
        arr2_shifted = np.roll(arr2, 1)
        result = np.fft.ifft(np.fft.fft(arr1) * np.fft.fft(arr2_shifted))
        result = np.round(result.real)
        return result.tolist()
    else:
        return cyclic_conv(arr2, arr1)


def main():
    # signal = [1, 2, 4, 6, 6, 7, 1, 4, 3, 10, 7, 5, 2, 3, 6, 5]
    # print(signal)
    # print(fft(signal))

    # Part one:

    # 4

    # mat = np.array([[f1(n, m) for m in range(64)] for n in range(32)])
    # plt.matshow(mat)
    # plt.title('original x[n,m]: ')
    # plt.colorbar()
    # plt.show()

    # dft = np.real(dft2d(mat))
    # print(dft)
    # plt.matshow(dft)
    # plt.title('DFT(x[n,m]): ')
    # plt.colorbar()
    # plt.show()

    # importing the images
    h = Image.open('h.png')
    y1 = Image.open('y1.png')
    y2 = Image.open('y2.png')
    y3 = Image.open('y3.png')

    mat_h = np.array(h)
    h0 = mat_h.transpose()[0]
    # print(h0)

    # 5

    # print("H(e^(j*0*n))", end=': ')
    # print(round_to(dtft(h0, 0), 10))
    # print("H(e^(j*pi/3*n))", end=': ')
    # print(round_to(dtft(h0, cm.pi / 3), 10))
    # print("H(e^(j*2pi/3*n))", end=': ')
    # print(round_to(dtft(h0, 2 * cm.pi / 3), 10))
    # print("H(e^(j*4pi/3A*n))", end=': ')
    # print(round_to(dtft(h0, 4 * cm.pi / 3), 10))

    # 6

    h0 = [0, 0, 23]
    w = [0 for _ in range(32)]
    w[0] = 1
    w[29] = 1
    # print(cyclic_conv(w, h0))
    # plt.stem(cyclic_conv(w, h0))
    # plt.title('The cyclic convolution of h0 and w: ')
    # plt.show()


if __name__ == "__main__":
    main()
