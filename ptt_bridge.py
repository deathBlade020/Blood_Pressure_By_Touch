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


def filter_ppg_savgol_dwt(ecg_signal):
    from scipy.signal import savgol_filter
    import pywt

    window_length = 15
    poly_order = 3
    unfiltered_data_smooth = savgol_filter(
        ecg_signal, window_length, poly_order)

    # print(ecg_signal)
    # print(unfiltered_data_smooth)

    # plt.subplot(2,1,1)
    plt.title("unfiltered")
    plt.plot(ecg_signal)

    # plt.subplot(2,1,2)
    # plt.title("filtered_ecg")
    # plt.plot(unfiltered_data_smooth)
    plt.show()

    return unfiltered_data_smooth


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

    ppg_path = "ppg_data/ppg_valid.xlsx"
    ecg_path = "ppg_data/ecg_valid.xlsx"

    ppg_df = pd.read_excel(ppg_path)
    ecg_df = pd.read_excel(ecg_path)

    ECG = []
    PPG = []
    BP = []

    import math
    for col in ppg_df.columns:
        tmp = ppg_df[col].tolist()
        bp = tmp[0].split("/")
        BP.append(int(bp[0]))
        ppg_array = [int(ele) for ele in tmp[1:] if not math.isnan(ele)]
        PPG.append(ppg_array)

    for col in ecg_df.columns:
        tmp = ecg_df[col].tolist()
        bp = tmp[0].split("/")
        ecg_array = [int(ele) for ele in tmp[1:] if not math.isnan(ele)]
        ECG.append(ecg_array)

    ic(len(ECG))
    ic(len(PPG))
    n_len = min(len(ECG), len(PPG))
    ravi = []

    # ppg_signal = np.array(PPG[0], dtype=np.float64)
    # ecg_signal = np.array(ECG[0], dtype=np.float64)
    rav_cnt = 0
    import time

    for i in range(n_len):
        ppg_signal = np.array(PPG[i], dtype=np.float64)
        ecg_signal = np.array(ECG[i], dtype=np.float64)

        diff_ecg = np.diff(ecg_signal)
        diff_ppg = np.diff(ppg_signal)

        average_ptt, r_peaks, systolic_peaks = calculate_ptt(
            ecg_signal, ppg_signal)

        plt.subplot(2, 1, 1)
        plt.title(BP[i])
        plt.plot(ecg_signal)
        # plt.scatter(r_peaks, [diff_ecg[i] for i in r_peaks], color="red")

        plt.subplot(2, 1, 2)
        plt.title(BP[i])
        plt.plot(ppg_signal)
        # plt.scatter(systolic_peaks, [diff_ppg[i]
        #             for i in systolic_peaks], color="red")
        # print(f"BP: {BP[i]}, PTT: {average_ptt:.2f}")
        ravi.append([BP[i], average_ptt])
        # if BP[i] == 118:
        #     plt.show()
        # plt.close()

        # plt.subplot(2, 1, 1)
        # plt.title(BP[i])
        # plt.plot(ecg_signal)

        # plt.subplot(2, 1, 2)
        # plt.title("PPG")
        # plt.plot(ppg_signal)
        # plt.savefig(f"save_here/plot_{i}.jpg")
        # plt.close()
        # print(f"index: {i} done")
        # plt.show()

        # if BP[i] >= 140:
        #     diff_ecg = np.diff(ecg_signal)
        #     diff_ppg = np.diff(ppg_signal)

        #     average_ptt, r_peaks, systolic_peaks = calculate_ptt(
        #         ecg_signal, ppg_signal)

        #     # print(f"BP: {BP[i]}, PTT: {average_ptt:.2f}")

        #     plt.subplot(2, 1, 1)
        #     plt.title(BP[i])
        #     plt.plot(diff_ecg)
        #     plt.scatter(r_peaks, [diff_ecg[i] for i in r_peaks], color="red")

        #     plt.subplot(2, 1, 2)
        #     plt.title(BP[i])
        #     plt.plot(diff_ppg)
        #     plt.scatter(systolic_peaks, [diff_ppg[i]
        #                 for i in systolic_peaks], color="red")

        #     # plt.show()

        #     rav_cnt += 1
        #     ravi.append([BP[i], average_ptt])

    ravi.sort(key=lambda x: x[0])
    # not_printed = 1

    for bp, ptt in ravi:
        print(f"BP: {bp}, PTT: {ptt:.2f}")
    print("END")


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
