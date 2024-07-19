import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read():

    filepath = "ptt_data\\Alisir4.xlsx"
    save = []
    df = pd.read_excel(filepath)
    for col in df.columns:
        arr = (df[col].tolist())
        arr = [ele for ele in arr]
        for ele in arr:
            print(ele, end=",")
        print("\n")

        # save.append(arr)

    plt.show()

import ctypes
def plot_stuff():
    import os
    import platform
    from icecream import ic

    extension = "so"
    if platform.system() == "windows":
        extension = "dll"
    
    os.system(f"gcc -shared -o peak_detection.dll -fPIC ptt.c")

    filepath = "ptt_data\\Alisir4.xlsx"
    ic(filepath)
    save = []
    df = pd.read_excel(filepath)
    for col in df.columns:
        arr = (df[col].tolist())
        arr = [ele for ele in arr]
        save.append(arr)

    if platform.system() == "Windows":
        lib = ctypes.CDLL("./peak_detection.dll")
    else:
        lib = ctypes.CDLL("./peak_detection.so")

    # Define the argument and return types for the C function
    lib.find_ptt.argtypes = [
        ctypes.POINTER(ctypes.c_double), 
        ctypes.POINTER(ctypes.c_double), 
        ctypes.c_int, 
        ctypes.POINTER(ctypes.c_int), 
        ctypes.POINTER(ctypes.c_int), 
        ctypes.POINTER(ctypes.c_int)
    ]
    lib.find_ptt.restype = ctypes.c_double
    MAX_PEAKS = 1000
    # Function to call the find_ptt function
    def find_ptt(ppg_signal_1, ppg_signal_2):
        n = len(ppg_signal_1)
        peak1 = np.zeros(MAX_PEAKS, dtype=np.int32)
        peak2 = np.zeros(MAX_PEAKS, dtype=np.int32)
        num_peaks = np.zeros(1, dtype=np.int32)

        ppg_signal_1_c = ppg_signal_1.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        ppg_signal_2_c = ppg_signal_2.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        peak1_c = peak1.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        peak2_c = peak2.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        num_peaks_c = num_peaks.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        average_ptt = lib.find_ptt(ppg_signal_1_c, ppg_signal_2_c, n, peak1_c, peak2_c, num_peaks_c)

        detected_peak1 = peak1[:num_peaks[0]]
        detected_peak2 = peak2[:num_peaks[0]]

        return average_ptt, detected_peak1, detected_peak2

    ppg_signal_1 = np.array(save[0], dtype=np.float64)
    ppg_signal_2 = np.array(save[1], dtype=np.float64)

    average_ptt, peaks1, peaks2 = find_ptt(ppg_signal_1, ppg_signal_2)

    print("PTT from python:", average_ptt)
    # print("Peaks in Signal 1:", peaks1)
    # print("Peaks in Signal 2:", peaks2)

    ppg_signal_1 = np.diff(ppg_signal_1)
    ppg_signal_2 = np.diff(ppg_signal_2)

    plt.subplot(2,1,1)
    plt.plot(ppg_signal_1)
    plt.scatter(peaks1, [ppg_signal_1[i] for i in peaks1], c='red')

    plt.subplot(2,1,2)
    plt.plot(ppg_signal_2)
    plt.scatter(peaks2, [ppg_signal_2[i] for i in peaks2], c='red')


    plt.show()


def stuff():
    for i in range(0,5):
        path = f"ptt_data\\output{i}.xlsx"
        print(path)
        save = []
        df = pd.read_excel(path)
        for col in df.columns:
            arr = (df[col].tolist())
            arr = [ele for ele in arr]
            save.append(arr)
        plt.subplot(2,1,1)
        plt.plot(save[1])
        plt.subplot(2,1,2)
        plt.plot(save[1])
        plt.show()
        print("end")


# read()
# stuff()
plot_stuff()
