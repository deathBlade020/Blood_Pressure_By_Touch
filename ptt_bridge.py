import scipy.signal as signal
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

    # print("PTT from python:", average_ptt)
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


def find_peaks(ppg_signal, threshold):
    peaks, _ = signal.find_peaks(ppg_signal, height=threshold)
    return peaks


def calculate_ptt_python(ecg_signal, ppg_signal, fs_ecg, fs_ppg):
    # Adjust threshold as needed
    r_peaks = find_peaks(ecg_signal, threshold=0.5)
    # Adjust threshold as needed
    systolic_peaks = find_peaks(ppg_signal, threshold=0.5)

    if len(r_peaks) == 0 or len(systolic_peaks) == 0:
        return None  # No valid peaks found

    ptt_values = []
    for r_peak in r_peaks:
        closest_systolic_peak = min(
            systolic_peaks, key=lambda x: abs(x - r_peak * fs_ppg / fs_ecg))
        ptt = abs(r_peak / fs_ecg - closest_systolic_peak / fs_ppg)
        ptt_values.append(ptt)

    return np.mean(ptt_values) if ptt_values else None


def normalise(data):
    max_val = np.max(data)
    min_val = np.min(data)
    new_data = [0] * len(data)
    for i in range(len(data)):
        new_data[i] = (data[i] - min_val)/(max_val - min_val)
    return new_data


def euclidean_distance(list1, list2):
    arr1 = np.array(list1)
    arr2 = np.array(list2)

    diff = arr1 - arr2

    distance = np.sqrt(np.sum(diff ** 2))

    return distance


def ecg_ppg_ptt():

    file_path = 'ptt_calculation.dll'

    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"File '{file_path}' has been deleted.")

    os.system("gcc -shared -o ptt_calculation.dll -fPIC ptt.c -lm")

    if platform.system() == "Windows":
        lib = ctypes.CDLL("./ptt_calculation.dll")
    else:
        lib = ctypes.CDLL("./ptt_calculation.so")

        lib.calculate_ptt.argtypes = [
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.c_int
        ]
    lib.calculate_ptt.restype = ctypes.c_double

    def calculate_ptt(ecg_signal, ppg_signal, bp):
        ecg_len = len(ecg_signal)
        ppg_len = len(ppg_signal)

        r_peaks = np.zeros(1000, dtype=np.int32)
        systolic_peaks = np.zeros(1000, dtype=np.int32)
        ecg_filt = np.zeros(ecg_len, dtype=np.double)
        ppg_filt = np.zeros(ppg_len, dtype=np.double)

        num_r_peaks = ctypes.c_int()
        num_systolic_peaks = ctypes.c_int()

        ecg_signal_c = ecg_signal.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double))
        ppg_signal_c = ppg_signal.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double))
        r_peaks_c = r_peaks.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        systolic_peaks_c = systolic_peaks.ctypes.data_as(
            ctypes.POINTER(ctypes.c_int))
        ecg_filt_c = ecg_filt.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        ppg_filt_c = ppg_filt.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        average_ptt = lib.calculate_ptt(
            ecg_signal_c, ppg_signal_c, ecg_len, ppg_len,
            r_peaks_c, systolic_peaks_c,
            ctypes.byref(num_r_peaks), ctypes.byref(num_systolic_peaks),
            ecg_filt_c, ppg_filt_c, bp
        )

        r_peaks = r_peaks[:num_r_peaks.value]
        systolic_peaks = systolic_peaks[:num_systolic_peaks.value]

        return average_ptt, r_peaks, systolic_peaks, ecg_filt, ppg_filt

    ecg_path = "ppg_data/ecg_valid_24July.xlsx"
    ppg_path = "ppg_data/ppg_valid_24July.xlsx"

    ppg_df = pd.read_excel(ppg_path)
    ecg_df = pd.read_excel(ecg_path)

    ECG = []
    PPG = []
    BP = []
    import math
    for col in ppg_df.columns:
        tmp = ppg_df[col].tolist()
        bp = tmp[0].split("/")
        BP.append([int(bp[0]), int(bp[1])])
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

    for i in range(n_len):
        bp_local = BP[i][0]
        # if bp_local > 100 and bp_local < 140:
        if bp_local >= 140:
            try:
                ppg_signal = np.array(PPG[i], dtype=np.float64)[0:130]
                ecg_signal = np.array(ECG[i], dtype=np.float64)[0:350]

                average_ptt, r_peaks, systolic_peaks, ecg_filt, ppg_filt = calculate_ptt(
                    ecg_signal, ppg_signal, bp_local)

                # r_peaks += 150

                ic(r_peaks)
                ic(systolic_peaks)
                print("\n")

                # plt.subplot(2, 1, 1)
                # plt.title(bp_local)
                # plt.plot(ecg_signal)
                # plt.scatter(r_peaks, [ecg_signal[i] for i in r_peaks], color='r')

                # plt.subplot(2, 1, 2)
                # plt.title(bp_local)

                # plt.plot(ppg_signal)
                # plt.scatter(systolic_peaks, [ppg_signal[i]
                #                              for i in systolic_peaks], color='r')
                # plt.show()

                # ic(bp_local,average_ptt)
                # print("\n")

                # print("\n")

                # ic(bp_local,average_ptt)
                ravi.append([bp_local, average_ptt])
            except Exception as err:
                print(err)

    ravi.sort(key=lambda x: x[0])
    # for bp, ptt in ravi:
    #     print(f"BP: {bp}, PTT: {ptt}")

    print("*** CODE ENDS HERE ***")


def plot_simple_graph(data, bp):
    plt.title(f"Blood Pressure: {bp}")
    plt.scatter(range(len(data)), data)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.show()


def read():
    ppg_path = "ppg_data/BP22July.xlsx"
    ecg_path = "ppg_data/ECG22July.xlsx"

    ppg_df = pd.read_excel(ppg_path)
    ecg_df = pd.read_excel(ecg_path)

    ECG = []
    PPG = []
    BP = []

    import math
    for col in ppg_df.columns:
        tmp = ppg_df[col].tolist()
        bp = tmp[0].split("/")
        BP.append([int(bp[0]), int(bp[1])])

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

    low = []
    high = []
    normal = []
    for i in range(n_len):
        if BP[i][0] < 100:
            low.append([PPG[i], BP[i][0]])
        elif BP[i][0] >= 100 and BP[i][0] < 140:
            normal.append([PPG[i], BP[i][0]])
        else:
            high.append([PPG[i], BP[i][0]])

    # for i in range(10):
    #     plot_simple_graph()

    # ic(high_bp_cnt)


# read()
# stuff()
# only_ppg_ptt()
ecg_ppg_ptt()
