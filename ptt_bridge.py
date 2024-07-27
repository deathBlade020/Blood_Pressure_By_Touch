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

    # ppg_signal = np.array(PPG[0], dtype=np.float64)
    # ecg_signal = np.array(ECG[0], dtype=np.float64)
    import time
    # problem = [10,28,35,41,43,45,47,48,49,50,51,54,56,57,59,61,62,65,66,68,71,72,73,74,76,83,85,86,88,90]
    problem = [10]

    till = 1
    import sys
    show = -1
    if len(sys.argv) < 2:
        show = 0
    else:
        show = int(sys.argv[1])
    check = 1

    if check:
        low = []
        low_bp = []
        normal = []
        normal_bp = []
        high = []
        high_bp = []
        limit = 10000
        for i in range(n_len):
            if i in [1,3,4,5,6,7,8,10,11,12]:
            # if i in [10,11,12]:


                # if BP[i][0] <= 100:
                ppg_signal = normalise(np.array(PPG[i], dtype=np.float64))
                # ppg_signal = normalise(ppg_signal)

                high.append(ppg_signal)
                limit = min(limit,len(ppg_signal))
                # ic(len(ppg_signal))
                high_bp.append(BP[i][0])

                # if BP[i][0] <= 100:
                #     low.append(ppg_signal)
                #     low_bp.append(BP[i][0])
                # elif BP[i][0] >= 101 and BP[i][0] < 140:
                #     normal.append(ppg_signal)
                #     normal_bp.append(BP[i][0])
                # else:
                    

        high_avg = [0] * limit

        for j in range(limit):
            for i in range(len(high)):
                high_avg[j] += high[i][j]

            # high_avg[j] /= limit
        ######################################################################
        ppg_path = "ppg_data/Test_Data.xlsx"

        check_df = pd.read_excel(ppg_path)
        check = []
        
        for col in check_df.columns:
            tmp = check_df[col].tolist()
            bp = tmp[0].split("/")
            ppg_array = [int(ele) for ele in tmp[1:] if not math.isnan(ele)]
            ppg_sig = ppg_array
            if len(ppg_sig) >= limit:
                tmp_arr = normalise(ppg_sig[0:limit])
                distance = euclidean_distance(high_avg, tmp_arr)
                ic(bp[0],distance)


       
        return
        ######################################################################
        # print(high_avg)

        for i in range(len(PPG)):
            if BP[i][0] >= 140:
                ppg_sig = PPG[i]
                if len(ppg_sig) >= limit:
                    tmp = PPG[i][0:limit]
                    tmp = normalise(tmp)
                    distance = euclidean_distance(high_avg, tmp)
                    ic(BP[i][0],distance)

        # plt.plot(high_avg)
        # plt.show()

        # min_length = min(min(len(low), len(normal)), len(high))
        # for i in range(min_length):

        #     LOW = np.array(low[i], dtype=np.float64)
        #     NORMAL = np.array(normal[i], dtype=np.float64)
        #     HIGH = np.array(high[i], dtype=np.float64)

        #     LOW = normalise(LOW)
        #     NORMAL = normalise(NORMAL)
        #     HIGH = normalise(HIGH)

        #     plt.subplot(3, 1, 1)
        #     plt.title(low_bp[i])
        #     # plt.scatter(list(range(len(HIGH))), HIGH,
        #     #             color='blue', marker='o', s=10)
        #     plt.plot(LOW)
        #     plt.gca().axes.get_xaxis().set_visible(False)

        #     plt.subplot(3, 1, 2)
        #     plt.title(normal_bp[i])
        #     # plt.scatter(list(range(len(NORMAL))), NORMAL,
        #     #             color='blue', marker='o', s=10)
        #     plt.plot(NORMAL)
        #     plt.gca().axes.get_xaxis().set_visible(False)

        #     plt.subplot(3, 1, 3)
        #     plt.title(high_bp[i])
        #     # plt.scatter(list(range(len(HIGH))), HIGH,
        #     #             color='blue', marker='o', s=10)
        #     plt.plot(HIGH)
        #     plt.gca().axes.get_xaxis().set_visible(False)

        #     plt.savefig(f"ppg_plots/plot_{i}.png")
        #     plt.close()
        # plt.show()

        return

    be = 0
    all_in = 1
    for i in range(n_len):
        if all_in:
            # if BP[i][0] >= 100 and BP[i][0] < 139:
            # if BP[i][0] <= 100:
            # if BP[i][0] >= 140:

            ppg_signal = np.array(PPG[i], dtype=np.float64)
            ecg_signal = np.array(ECG[i], dtype=np.float64)
            mean = math.ceil(np.mean(ppg_signal))
            median = np.median(ppg_signal)
            max_ppg = np.max(ppg_signal)
            min_ppg = np.min(ppg_signal)

            diff = abs(mean - median)
            # ic(BP[i][0], mean, median, max_ppg, min_ppg)

            # average_ptt, r_peaks, systolic_peaks, ecg_filt, ppg_filt = calculate_ptt(
            #     ecg_signal, ppg_signal, BP[i][0])

            # plt.subplot(2, 1, 1)
            # plt.title(BP[i][0])
            # plt.plot(ecg_signal)
            # plt.scatter(r_peaks, [ecg_signal[i] for i in r_peaks], color='r')

            # plt.subplot(2, 1, 2)
            # plt.title(BP[i][0])

            # plt.plot(ppg_signal)
            # plt.scatter(systolic_peaks, [ppg_signal[i]
            #                              for i in systolic_peaks], color='r')

            # if show:
            #     plt.show()
            #     plt.close()
            # else:
            #     plt.savefig(f"ppg_plots/plot_{i}.png")

            # break

    # ravi.sort(key=lambda x: x[0])
    # not_printed = 1

    # for bp, ptt in ravi:
    #     print(f"BP: {bp[0]}/{bp[1]}, ptt: {ptt:.2f}")

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
