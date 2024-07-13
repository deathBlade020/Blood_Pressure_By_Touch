import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from icecream import ic
import joblib
import json
import os
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
import math
import pandas as pd
from scipy.stats import uniform, randint
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from icecream import ic
import warnings
warnings.filterwarnings("ignore")


def butter_highpass(cutoff, fs, order):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def highpass_filter(data, cutoff=0.3, fs=25, order=4):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def bandpass_filter(ppg_signal, lowcut=0.5, highcut=5.0, fs=25, order=2):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, ppg_signal)
    return y


def plot_multi(ppg, ppg_bwr):
    plt.subplot(2, 1, 1)
    plt.plot(ppg)
    plt.subplot(2, 1, 2)
    plt.plot(ppg_bwr)
    plt.show()


def plot_single(ppg):
    plt.plot(ppg)
    plt.show()


def get_column_labels():
    def generate_column_labels(start, end):
        labels = []
        current = start
        while current != end:
            labels.append(current.upper())
            current = increment_column_label(current)
        labels.append(end.upper())
        return labels

    def increment_column_label(label):
        if label[-1] != 'z':
            return label[:-1] + chr(ord(label[-1]) + 1)
        else:
            return increment_column_label(label[:-1]) + 'a' if label[:-1] else 'aa'

    start_label = 'jc'
    end_label = 'oj'
    column_labels = generate_column_labels(start_label, end_label)
    print(column_labels)


def find_heart_rate(sys_peaks, fs):
    if len(sys_peaks) == 0:
        return 0
    ic(np.diff(sys_peaks))

    peak_intervals = np.diff(sys_peaks) / fs
    ic(peak_intervals)
    heart_rates = [60 / val for val in peak_intervals]
    ic(heart_rates)

    average_heart_rate = np.mean(heart_rates)
    # ic(average_heart_rate)
    return average_heart_rate


def find_refliction_index(ppg_bwr, sys_peaks):
    n = len(ppg_bwr)
    inflection_peak = []
    tol = 1
    for peak in sys_peaks:
        idx = peak
        while idx+1 < n and ppg_bwr[idx] >= ppg_bwr[idx+1]:
            idx += 1
        # print(f"before: {peak}, after: {idx}")

        if idx != peak and abs(idx - peak) >= tol:
            inflection_peak.append(idx)

    len_sys_peak = len(sys_peaks)
    if len_sys_peak == 0:
        return 0

    numerator, denominator = 0, 0
    for i in range(len_sys_peak):
        numerator += ppg_bwr[inflection_peak[i]]
        denominator += ppg_bwr[sys_peaks[i]]

    refliction_index = numerator/denominator
    return refliction_index


def find_lasi(ppg, fs):
    peaks = find_sys_foot_peaks(ppg,fs)
    sys,foot = [],[]
    for s,d in peaks:
        sys.append(s)
        foot.append(d)   
    # ic(sys)
    ic(foot)

    s3 = []
    for i in range(len(sys)-1):
        half_pos = int((sys[i] + foot[i+1])/2)
        # ic(sys[i], half_pos)
        s3.append(half_pos)

    s2 = sys
    sampling_intrval = 1/fs
    min_length = min(len(s2),len(s3))
    area = 0
    for i in range(min_length):
        s2_index = s2[i]
        s3_index = s3[i]
        area += trapezoidal_area(ppg,s2_index,s3_index,sampling_intrval)
    
    area /= min_length
    ic(area)
    return area



def plot_scatter(data, peaks, mean=None, std=None):
    plt.plot(data)
    if mean != None and std != None:
        plt.title(str(mean) + " = " + str(std))
    plt.scatter(peaks, [data[i] for i in peaks], c='red')
    plt.show()


def find_sys_peaks(ppg_bwr, fs):
    sys_peaks, _ = find_peaks(ppg_bwr, prominence=0.6, distance=10)
    return sys_peaks


def calculate_pulse_areas(ppg_signal, fs):

    inverted_ppg_signal = -ppg_signal
    valleys, _ = find_peaks(inverted_ppg_signal, distance=10)
    # ic(valleys)
    # plot_scatter(ppg_signal,valleys)
    valleys = valleys[1:]
    A1, A2 = [], []
    for i in range(len(valleys)-1):
        start_index = i
        end_index = i+1
        pulse_segment = ppg_signal[start_index:end_index+1]
        time_segment = np.arange(start_index, end_index+1) / fs
        pulse_area = np.trapz(pulse_segment, time_segment)
        # ic(valleys[i],valleys[i+1],pulse_area)
        A1.append(pulse_area * 0.75)
        A2.append(pulse_area * 0.25)

    # ic(A1)
    # ic(A2)

    avg_a1 = sum(A1)/len(A1)
    avg_a2 = sum(A2)/len(A2)

    # ic(avg_a2/avg_a1)
    return avg_a2/avg_a1


def find_mnpv(ppg_signal, fs):
    sys_peaks, _ = find_peaks(ppg_signal, prominence=0.3, distance=15)
    iac = 0
    div = 0
    for i in range(1, len(sys_peaks)):
        iac += (ppg_signal[i] - ppg_signal[i-1])/fs
        div += 1
    if div == 0:
        return 0
    iac /= div
    peak_intervals = np.diff(sys_peaks) / fs
    average_interval = np.mean(peak_intervals)
    return iac/(iac + average_interval)


def find_crest_time(ppg_signal, fs):
    sys_peaks, _ = find_peaks(ppg_signal, prominence=0.3, distance=15)
    inverted_ppg_signal = -ppg_signal
    valleys, _ = find_peaks(inverted_ppg_signal, distance=10)
    crest_time = 0
    div = 0
    for a, b in zip(sys_peaks, valleys):
        # ic(a, b)
        div += 1
        crest_time += (a-b)/fs
    if div == 0:
        return 0
    crest_time /= div
    return crest_time


def find_sys_foot_peaks(ppg, fs):
    sys_peaks = find_sys_peaks(ppg, fs)
    PEAKS = []
    n = len(ppg)
    # store = []
    for peak in sys_peaks:
        idx = peak
        while idx + 1 < n and ppg[idx] >= ppg[idx+1]:
            idx +=1
        PEAKS.append([peak, idx])
        # store.append(idx)
    # plot_scatter(ppg,store)

    return PEAKS


def find_both_times(ppg, fs):
    PEAKS = find_sys_foot_peaks(ppg, fs)
    sys_time, foot_time = 0, 0
    peaks_len = len(PEAKS)

    if peaks_len == 0:
        return sys_time, foot_time

    for i in range(peaks_len):
        sys_time += ((PEAKS[i][0] - PEAKS[i][1])/fs)
    sys_time /= peaks_len

    for i in range(peaks_len - 1):
        foot_time += ((PEAKS[i+1][1] - PEAKS[i][0])/fs)
    foot_time /= peaks_len
    return sys_time, foot_time


def find_pir(ppg, fs):
    new_ppg = [-1 * item for item in ppg]
    PEAKS = find_sys_foot_peaks(new_ppg, fs)
    n = len(PEAKS)
    if n == 0:
        return 0
    # ic(PEAKS)
    pir = 0
    for s, d in PEAKS:
        pir += new_ppg[s]/new_ppg[d]
    pir /= n
    return pir


def find_augmentation_index(ppg, fs):
    peaks = find_sys_peaks(ppg, fs)
    primary_peak = ppg[peaks[0]]
    reflected_peak = ppg[peaks[1]]
    augmentation_pressure = reflected_peak - primary_peak
    pulse_pressure = np.max(ppg) - np.min(ppg)
    aix = augmentation_pressure / pulse_pressure
    return aix


def find_pulse_height(ppg, fs):
    PEAKS = find_sys_foot_peaks(ppg, fs)
    len_peaks = len(PEAKS)
    if len_peaks == 0:
        return 0
    sys_peaks, foot_peaks = [], []
    pulse_height = 0
    for a, b in PEAKS:
        sys_peaks.append(a)
        foot_peaks.append(b)
        pulse_height += ppg[a] - ppg[b]
    pulse_height /= len_peaks
    return pulse_height

def find_pulse_width(ppg, fs):
    PEAKS = find_sys_foot_peaks(ppg, fs)
    len_peaks = len(PEAKS)
    if len_peaks == 0:
        return 0
    A = [ele[0] for ele in PEAKS]
    B = [ele[1] for ele in PEAKS]
    pulse_width = []
    for i in range(len(B)-1):
        width = (B[i] - B[i+1])/fs
        pulse_width.append(width)

    return np.mean(pulse_width)

def find_hrv(ppg,fs):
    sys_peaks = find_sys_peaks(ppg,fs)
    arr = [ppg[peak] for peak in sys_peaks]
    ppi = np.diff(arr)
    hrv = np.std(ppi)
    return hrv

def find_amplitude_ratios(ppg,fs):
    sys_peaks = find_sys_peaks(ppg,fs)
    amp_arr = [ppg[peak] for peak in sys_peaks]
    amplitude_ratios = amp_arr / np.roll(amp_arr, 1)
    amplitude_ratios = amplitude_ratios[1:]
    mean_amplitude_ratios = np.mean(amplitude_ratios)
    return mean_amplitude_ratios

def find_max_min_amplitudes(ppg):
    max_amplitude = max(ppg)
    min_amplitude = min(ppg)
    return max_amplitude,min_amplitude

def find_womersley_number(ppg,fs):
    foot_peaks = find_sys_foot_peaks(ppg,fs)
    amp = [ppg[ele[1]] for ele in foot_peaks]
    return np.mean(amp)

def find_alpha(ppg,fs):
    peaks = find_sys_foot_peaks(ppg, fs)
    sys_peaks = [ele[0] for ele in peaks]
    foot_peaks = [ele[1] for ele in peaks]
    alpha = 0
    for i in range(len(sys_peaks)-1):
        alpha += (sys_peaks[i] - foot_peaks[i] - foot_peaks[i+1])
    alpha /= len(sys_peaks)
    return alpha

def trapezoidal_area(ppg, start_index, end_index, sampling_interval):
    # ic(sampling_interval)
    segment = ppg[start_index:end_index+1]
    # ic(start_index,end_index,segment)

    area = 0.0
    for i in range(len(segment) - 1):
        area += (segment[i] + segment[i+1]) / 2 * sampling_interval
    return area





def find_ipa(ppg,fs):
    peaks = find_sys_foot_peaks(ppg,fs)
    sys,foot = [],[]
    s1 = []
    tol = 0
    for s,d in peaks:
        sys.append(s)
        foot.append(d)   
        half_neg = int((s+d)/2)
        half_neg += tol
        s1.append(half_neg + tol)

    s3 = [int((sys[i] + foot[i+1])/2) for i in range(len(sys)-1)]
    

    s4 = foot[1:]
    s2 = sys


    sampling_interval = 1/fs
    min_length = min(len(s1),len(s2),len(s3),len(s4))
    AREA = 0
    total = 0
    for i in range(min_length):
        s1_index = s1[i]
        s2_index = s2[i]
        s3_index = s3[i]
        s4_index = s4[i]
        ic(s1_index,s2_index,s3_index,s4_index)

        area1 = trapezoidal_area(ppg, s2_index, s1_index, sampling_interval)
        area2 = trapezoidal_area(ppg, s3_index, s4_index, sampling_interval)
        ipa_ratio = area1 / area2 if area2 != 0 else np.inf
        if ipa_ratio:
            AREA += ipa_ratio
            total += 1

    return AREA

def find_systolic_time_x(ppg, fs,val = 20):
    ppg = ppg[::-1]
    sys_peaks = find_sys_peaks(ppg, fs)
    # plot_scatter(ppg,sys_peaks)
    peak_len = len(sys_peaks)
    if peak_len == 0:
        return 0
    
    val += 15
    mul_fact = val/100
    ic(mul_fact)
    foot = []
    sys_time = 0
    div = 0
    store = []
    for peak in sys_peaks:
        idx = peak
        while idx >= 1 and ppg[idx] >= ppg[idx - 1]:
            idx -= 1
        # add = math.ceil(idx + mul_fact * (peak - idx))
        add = idx + math.ceil(mul_fact * abs(peak - idx))
        ic(idx, add)
        # print(f"peak: {ppg[peak]}, 10%: {ppg[add]}")
        foot.append(add)
        if peak != add:
            store.append(add)
            div+=1
            sys_time += ((ppg[peak] - ppg[add])/fs)

    # plot_scatter(ppg,store)
    sys_time /= div
    # ic(sys_time)
    return sys_time

def find_pwv(ppg,fs):
    ppg = ppg[::-1]
    sys_peaks = find_sys_peaks(ppg, fs)

    peak_len = len(sys_peaks)
    if peak_len == 0:
        return 0
    foot = []
    for peak in sys_peaks:
        idx = peak
        while idx >= 1 and ppg[idx] >= ppg[idx - 1]:
            idx -= 1
        foot.append(idx)
    sampling_rate = 1/fs
    time_add = 0
    for i in range(peak_len):
        time_add += ((sys_peaks[i] - foot[i])/fs)
    distance = 1
    pwv = distance / time_add
    # ic(pwv)
    # plot_scatter(ppg,foot)
    return pwv

def extract_features(ppg, fs=25):

    ppg_filtered = bandpass_filter(ppg)
    sys_peaks = find_sys_peaks(ppg_filtered, fs)
    hr = find_heart_rate(sys_peaks, fs)

    ref_ind = find_refliction_index(ppg_filtered, sys_peaks)
    lasi = find_lasi(ppg, fs)
    crest_time = find_crest_time(ppg_filtered, fs)
    mnpv = find_mnpv(ppg_filtered, fs)
    sys_time, foot_time = find_both_times(ppg, fs)
    # ic(sys_time,foot_time)

    pir = find_pir(ppg, fs)

    aix = find_augmentation_index(ppg, fs)
    pulse_height = find_pulse_height(ppg, fs)
    pulse_width = find_pulse_width(ppg, fs)
    
    hrv = find_hrv(ppg,fs)
    # ic(hrv)

    amplitude_ratios = find_amplitude_ratios(ppg,fs)

    max_amplitude, min_amplitude = find_max_min_amplitudes(ppg)
    # ic(max_amplitude)
    # ic(min_amplitude)

    womersley_number = find_womersley_number(ppg,fs)
    
    alpha = find_alpha(ppg_filtered,fs)
    
    ipa = find_ipa(ppg, fs)

    sys_time_ten = find_systolic_time_x(ppg, fs)
    ic(sys_time_ten)

    pwv = find_pwv(ppg,fs)



def main():
    data = [ 123.0, 128.0, 130.0, 128.0, 128.0, 130.0, 131.0, 133.0, 134.0, 133.0, 121.0, 117.0, 120.0, 124.0, 128.0, 130.0, 127.0, 127.0, 128.0, 133.0, 134.0, 135.0, 131.0, 120.0, 119.0, 121.0, 126.0, 130.0, 131.0, 130.0, 130.0, 131.0, 133.0, 135.0, 137.0, 133.0, 121.0, 120.0, 123.0, 128.0, 131.0, 131.0, 130.0, 131.0, 133.0, 134.0, 137.0, 138.0, 134.0, 123.0, 120.0, 123.0, 127.0, 131.0, 133.0, 131.0, 130.0, 131.0, 133.0, 134.0, 137.0, 135.0, 126.0, 121.0, 124.0, 128.0, 133.0, 134.0, 133.0, 131.0, 131.0, 133.0, 135.0, 137.0, 138.0, 133.0, 123.0, 121.0, 124.0, 128.0, 133.0, 133.0, 131.0, 131.0, 131.0, 134.0, 137.0, 140.0, 138.0, 127.0, 123.0, 123.0, 127.0, 131.0, 134.0, 133.0, 130.0, 131.0, 133.0, 135.0, 138.0, 141.0, 137.0, 124.0, 123.0, 124.0, 128.0, 133.0, 134.0, 134.0, 133.0, 134.0, 135.0, 138.0, 140.0, 141.0, 134.0, 124.0, 123.0, 126.0, 130.0, 133.0, 134.0, 133.0, 133.0, 134.0, 137.0, 138.0, 141.0, 140.0, 128.0, 123.0, 123.0, 127.0, 131.0, 135.0, 134.0, 131.0, 131.0, 133.0, 135.0, 138.0, 140.0, 137.0, 126.0, 123.0, 124.0, 128.0, 133.0, 134]
    peaks = [8,11,21,24,34,37,47,50,60,63,74,77,87,91,101,104,115,118,128,132,142,145]
    # plot_scatter(data,peaks)

    fs = 25
    extract_features(data,fs)

    # os.system("gcc -o ravi find_systolic_peaks.c findpeaks.c string1.c rtGetInf.c rt_nonfinite.c sort1.c sortIdx.c rtGetNaN.c eml_setop.c")
    # os.system("ravi.exe")


main()
