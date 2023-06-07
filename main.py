import numpy as np
import math


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