import numpy as np
import math
from PIL import Image
import cmath as cm
import matplotlib.pyplot as plt
import simpleaudio as sa
import wave


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


def idft2d(mat):
    mat = np.array(mat).transpose()
    m = np.array([np.fft.ifft(mat[i]) for i in range(len(mat))]).transpose()
    return np.array([np.fft.ifft(m[i]) for i in range(len(m))])


def cyclic_conv(arr1, arr2):
    if len(arr1) >= len(arr2):
        arr2 += [0] * (len(arr1) - len(arr2))
        arr2_shifted = np.roll(arr2, 1)
        result = np.fft.ifft(np.fft.fft(arr1) * np.fft.fft(arr2_shifted))
        result = np.round(result.real)
        return result.tolist()
    else:
        return cyclic_conv(arr2, arr1)


def sample_signal(filename, fs):
    with wave.open(filename, "rb") as wf:
        num_samples = wf.getnframes()
        signal = np.frombuffer(wf.readframes(num_samples), dtype=np.int16)
    return signal[::fs]


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
    # h = Image.open('h.png')
    # y1 = Image.open('y1.png')
    # y2 = Image.open('y2.png')
    # y3 = Image.open('y3.png')

    # mat_h = np.array(h)
    # h0 = mat_h.transpose()[0]
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

    # h0 = [0, 0, 23]
    # w = [0 for _ in range(32)]
    # w[0] = 1
    # w[29] = 1
    # print(cyclic_conv(w, h0))
    # plt.stem(cyclic_conv(w, h0))
    # plt.title('The cyclic convolution of h0 and w: ')
    # plt.show()

    # 7

    # Y1 = dft2d(np.array(y1))
    # h1 = np.zeros(Y1.shape)
    # h1[0:np.array(h).shape[0], 0:np.array(h).shape[1]] = h
    # H1 = dft2d(h1)
    # X1 = Y1 / H1

    # plt.matshow(np.real(X1), cmap='cividis')
    # plt.title('X1[n,m]: ')
    # plt.show()

    # x1 = idft2d(X1)
    # plt.matshow(np.real(x1), cmap='cividis')
    # plt.title('Recovered x1[n,m]: ')
    # plt.show()

    # Y2 = dft2d(np.array(y2))
    # h2 = np.zeros(Y2.shape)
    # h2[0:np.array(h).shape[0], 0:np.array(h).shape[1]] = h
    # H2 = dft2d(h2)
    # X2 = Y2 / H2

    # plt.matshow(np.real(X2), cmap='cividis')
    # plt.title('X2[n,m]: ')
    # plt.show()

    # x2 = idft2d(X2)
    # plt.matshow(np.real(x2), cmap='cividis')
    # plt.title('Recovered x2[n,m]: ')
    # plt.show()

    # Part two

    d1 = 0.214609687
    d2 = 0.325694081
    d = (d1 + d2) % 0.5
    N = 2 ** 16

    fs = 16000
    note = sample_signal("test.wav", 6)
    x = np.pad(note / max(note), int((N - len(note)) / 2))
    x2 = [k ** 2 for k in x]

    # 1

    p = sum(x2) / N
    # print(p)

    # 2

    w1 = 1.6 + 0.1 * d1
    w2 = 1.6 + 0.1 * d
    w3 = 3
    z = [50 * np.sqrt(p) * (np.cos(w1 * n) + np.cos(w2 * n) + np.cos(w3 * n)) for n in range(N)]
    y = x + z
    # audio = y.astype(np.int16)
    # play_obj = sa.play_buffer(audio, 1, 2, fs)
    # play_obj.wait_done()

    # 3

    # plt.plot(y)
    # plt.title('y[n]:')
    # plt.show()

    # 4

    # Y = [dtft(y, (i - 128) * 2 * np.pi / 128) for i in range(257)]
    # plt.plot(Y)
    # plt.title('Y[k]:')
    # plt.show()

    # 6

    # y2 = y[::2]
    # plt.plot(y2)
    # plt.title('y2[n]:')
    # plt.show()

    # Y2 = [dtft(y2, (i - 128) * 2 * np.pi / 128) for i in range(257)]
    # plt.plot(Y2)
    # plt.title('Y2[k]:')
    # plt.show()

    x = np.pad(note, int((N - len(note)) / 2))
    y = x + z
    # audio = y[::2].astype(np.int16)
    # play_obj = sa.play_buffer(audio, 1, 2, fs)
    # play_obj.wait_done()


if __name__ == "__main__":
    main()
