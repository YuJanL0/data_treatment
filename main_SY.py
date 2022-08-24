import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fftpack import fft, fftfreq, rfft, rfftfreq


def main():
    df = pd.read_excel(input("Filename: ")+".xlsx", header=None)
    Fs = 1
    plt.figure(figsize=(10, 4))
    plt.plot(df[0])
    plt.plot(df[1])
    plt.show()
    t = input("Interval 'start/end' or 'a' for all: ")
    if t == 'a':
        pass
    else:
        t = t.split('/')
        s = int(t[0]) - 1
        e = int(t[1]) - 1
        df = df.loc[s:e, :]
        print(df)
    plt.plot(df[0])
    plt.plot(df[1])
    plt.show()
    for i in range(2):
        y = np.array(df[i])
        peaks, _ = find_peaks(y, distance=len(y)/100)
        valleys, _ = find_peaks(-y, distance=len(y)/100)
        print(peaks, valleys)
        start = min(np.min(peaks), np.min(valleys))
        if np.min(peaks) < np.min(valleys):
            end = np.max(peaks)
            if np.max(peaks) <= np.max(valleys):
                valleys = valleys[:-1]
        else:
            end = np.max(valleys)
            if np.max(valleys) <= np.max(peaks):
                peaks = peaks[:-1]
        peaks -= start
        valleys -= start
        y = y[start:end + 1]
        peaks_value = y[peaks]
        valleys_value = y[valleys]
        x = np.arange(len(y))
        plt.figure(figsize=(10, 4))
        plt.plot(x, y)
        plt.scatter(x[peaks], y[peaks])
        plt.scatter(x[valleys], y[valleys])
        plt.title("data " + str(i))
        plt.show()

        amplitudes = []
        for j in range(min(len(peaks_value), len(valleys_value))):
            amplitudes.append((peaks_value[j] - valleys_value[j]) / 2)
        print("data " + str(i) + ": amplitudes = " + str(amplitudes))

        periods = []
        for j in range(len(peaks_value) - 1):
            periods.append(peaks[j + 1] - peaks[j])
        print("data " + str(i) + ": periods = " + str(periods))

        y_fft = fft(y)
        amplitudes = 2 / len(y) * np.abs(y_fft)

        frequencies = fftfreq(len(y)) * len(y) * Fs
        plt.plot(frequencies[1:len(frequencies) // 2], amplitudes[1:len(y_fft) // 2])
        print("data " + str(i) + ": amplitude from fft: " + str(max(amplitudes[1:len(y_fft) // 2])))
        plt.show()

        y_ = rfft(y, n=len(y))
        a = np.abs(y_)[1:]
        freqs = rfftfreq(len(y), d=1. / 60)[1:]
        freqs = np.divide(60, freqs)
        max_freq = freqs[np.argmax(a)]
        print("data " + str(i) + ": period from fft: " + str(max_freq))
        plt.plot(freqs, a)
        plt.show()


if __name__ == '__main__':
    main()

