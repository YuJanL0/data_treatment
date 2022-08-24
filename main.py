import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fftpack import fft, fftfreq, rfft, rfftfreq


def main():
    df_ITF = pd.read_excel("10_1.xls", usecols=[0, 1], header=None)
    df_ITF.columns = ["time", "ITF_pos"]
    df_vol = pd.read_excel("10_1.xls", usecols=[2, 3], header=None)
    df_vol.columns = ["time", "voltage"]
    df_ITF.set_index("time", inplace=True)
    df_vol.set_index("time", inplace=True)
    print(df_ITF)
    print(df_vol)

    Fs = 1
    plt.figure(figsize=(10, 4))
    plt.plot(df_ITF)
    plt.plot(df_vol)
    plt.show()

    t = input("Interval 'start/end' or 'a' for all: ")
    if t == 'a':
        pass
    else:
        t = t.split('/')
        s = float(t[0])
        e = float(t[1])
        df_ITF = df_ITF.loc[s:e, :]
        df_vol = df_vol.loc[s:e, :]
        print(df_ITF)
        print(df_vol)

    plt.plot(df_ITF)
    plt.plot(df_vol)
    plt.show()

    for i in range(1):
        peaks, _ = find_peaks(df_ITF['ITF_pos'], distance=len(df_ITF) / 1000)
        valleys, _ = find_peaks(-df_ITF['ITF_pos'], distance=len(df_ITF) / 1000)
        print(peaks, valleys)

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
        print("Amplitudes = " + str(amplitudes))

        periods = []
        for j in range(len(peaks_value) - 1):
            periods.append(peaks[j + 1] - peaks[j])
        print("Periods = " + str(periods))


if __name__ == '__main__':
    main()

