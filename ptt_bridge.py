import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ctypes
import os
import platform
from icecream import ic
import json

# def read():

#     filepath = "ptt_data\\Alisir4.xlsx"
#     save = []
#     df = pd.read_excel(filepath)
#     for col in df.columns:
#         arr = (df[col].tolist())
#         arr = [ele for ele in arr]
#         for ele in arr:
#             print(ele, end=",")
#         print("\n")

#         # save.append(arr)

#     plt.show()


def only_ppg_ptt():

    extension = "so"
    if platform.system() == "windows":
        extension = "dll"

    os.system("gcc -shared -o peak_detection.dll -fPIC ptt.c")

    filepath = "ppg_data\\ayush1.xlsx"
    # filepath = "ptt_data\\Alisir2.xlsx"

    ic(filepath)
    save = []
    df = pd.read_excel(filepath)
    for col in df.columns:
        arr = df[col].tolist()
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

        ppg_signal_1_c = ppg_signal_1.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double))
        ppg_signal_2_c = ppg_signal_2.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double))
        peak1_c = peak1.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        peak2_c = peak2.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        num_peaks_c = num_peaks.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        average_ptt = lib.find_ptt(
            ppg_signal_1_c, ppg_signal_2_c, n, peak1_c, peak2_c, num_peaks_c)

        detected_peak1 = peak1[:num_peaks[0]]
        detected_peak2 = peak2[:num_peaks[0]]

        return average_ptt, detected_peak1, detected_peak2

    ppg_signal_1 = np.array(save[0], dtype=np.float64)
    ppg_signal_2 = np.array(save[1], dtype=np.float64)

    ppg_signal_1 = ppg_signal_1[50:]
    ppg_signal_2 = ppg_signal_2[50:]

    average_ptt, peaks1, peaks2 = find_ptt(ppg_signal_1, ppg_signal_2)

    print("PTT from python:", average_ptt)
    # print("Peaks in Signal 1:", peaks1)
    # print("Peaks in Signal 2:", peaks2)

    ppg_signal_1 = np.diff(ppg_signal_1)
    ppg_signal_2 = np.diff(ppg_signal_2)

    plt.subplot(2, 1, 1)
    plt.plot(ppg_signal_1)
    plt.scatter(peaks1, [ppg_signal_1[i] for i in peaks1], c='red')

    plt.subplot(2, 1, 2)
    plt.plot(ppg_signal_2)
    plt.scatter(peaks2, [ppg_signal_2[i] for i in peaks2], c='red')

    plt.show()


def stuff():
    for i in range(0, 5):
        path = f"ptt_data\\output{i}.xlsx"
        print(path)
        save = []
        df = pd.read_excel(path)
        for col in df.columns:
            arr = (df[col].tolist())
            arr = [ele for ele in arr]
            save.append(arr)
        plt.subplot(2, 1, 1)
        plt.plot(save[1])
        plt.subplot(2, 1, 2)
        plt.plot(save[1])
        plt.show()
        print("end")


def ecg_ppg_ptt():

    os.system("gcc -shared -o ptt_calculation.dll -fPIC ptt.c -lm")

    if platform.system() == "Windows":
        lib = ctypes.CDLL("./ptt_calculation.dll")
    else:
        lib = ctypes.CDLL("./ptt_calculation.so")

    lib.calculate_ptt.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)
    ]
    lib.calculate_ptt.restype = ctypes.c_double

    def calculate_ptt(ecg_signal, ppg_signal):
        ecg_len = len(ecg_signal)
        ppg_len = len(ppg_signal)
        r_peaks = np.zeros(1000, dtype=np.int32)
        systolic_peaks = np.zeros(1000, dtype=np.int32)
        num_r_peaks = ctypes.c_int()
        num_systolic_peaks = ctypes.c_int()

        ecg_signal_c = ecg_signal.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double))
        ppg_signal_c = ppg_signal.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double))
        r_peaks_c = r_peaks.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        systolic_peaks_c = systolic_peaks.ctypes.data_as(
            ctypes.POINTER(ctypes.c_int))

        average_ptt = lib.calculate_ptt(
            ecg_signal_c, ppg_signal_c, ecg_len, ppg_len,
            r_peaks_c, systolic_peaks_c,
            ctypes.byref(num_r_peaks), ctypes.byref(num_systolic_peaks)
        )

        r_peaks = r_peaks[:num_r_peaks.value]
        systolic_peaks = systolic_peaks[:num_systolic_peaks.value]

        return average_ptt, r_peaks, systolic_peaks

    filepath_ecg = "ecg_data\\ecgRaw70json.json"
    ic(filepath_ecg)

    with open(filepath_ecg, "r") as handle:
        data = json.load(handle)
        sig = np.array(data[0]["lead1"], dtype=np.float64)
        ecg_signal = np.array([-1 * ele for ele in sig])
        # plt.plot(ecg_signal)
        # plt.show()
    # return
    ecg_signal = ecg_signal[50:]

    filepath_ppg = "ppg_data\\Alisir4.xlsx"
    ic(filepath_ppg)

    save = []
    df = pd.read_excel(filepath_ppg)
    for col in df.columns:
        arr = df[col].tolist()
        arr = [ele for ele in arr]
        save.append(arr)

    ppg_signal = np.array(save[0][50:], dtype=np.float64)

    diff_ecg = np.diff(ecg_signal)
    diff_ppg = np.diff(ppg_signal)

    average_ptt, r_peaks, systolic_peaks = calculate_ptt(
        ecg_signal, ppg_signal)

    print("PTT PYTHON:", average_ptt)
    # ic(len(r_peaks))
    # ic(len(systolic_peaks))

    # print("R Peaks:", r_peaks)
    # print("Systolic Peaks:", systolic_peaks)

    plt.subplot(2, 1, 1)
    plt.plot(diff_ecg)
    plt.title("ECG")
    plt.scatter(r_peaks, [diff_ecg[i] for i in r_peaks], c="red")

    plt.subplot(2, 1, 2)
    plt.plot(diff_ppg)
    plt.title("PPG")
    plt.scatter(systolic_peaks, [diff_ppg[i] for i in systolic_peaks], c="red")

    # plt.show()


def read():
    filepath = "ecg_data\\ecgs_pvc.json"
    with open(filepath, "r") as handle:
        data = json.load(handle)
        sig = np.array(data[0]["lead1"], dtype=np.float64)
        sig = [-1 * ele for ele in sig]
        plt.plot(sig)
        plt.show()


# read()
# stuff()

# only_ppg_ptt()
ecg_ppg_ptt()
# read()
