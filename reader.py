import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
from icecream import ic
from scipy.signal import butter, filtfilt, find_peaks
import pandas as pd
import numpy as np
import math
from scipy.signal import butter, filtfilt, find_peaks
from icecream import ic
from scipy.signal import argrelextrema

from numpy.polynomial import Polynomial
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
    peak_intervals = np.diff(sys_peaks) / fs
    heart_rates = 60 / peak_intervals
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
        # print(f"Peak: {peak}, index: {idx}")
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
    ic(sys)
    ic(foot)
    s3 = []
    for i in range(len(sys)-1):
        half_pos = int((sys[i] + foot[i+1])/2)
        s3.append(half_pos)
    ic(s3)
    s2 = sys
    sampling_intrval = 1/fs
    min_length = min(len(s2),len(s3))
    area = 0
    for i in range(min_length):
        s2_index = s2[i]
        s3_index = s3[i]
        area += trapezoidal_area(ppg,min(s2_index,s3_index),max(s2_index,s3_index),sampling_intrval)
    area /= min_length
    return area



def plot_scatter(data, peaks, close=False):
    plt.plot(data, label='max_slopes')
    plt.scatter(peaks, [data[i] for i in peaks], c='red',label='peaks')
    plt.title("max_slopes") 
    
    if close:
        plt.legend()
        plt.show(block=False)
        plt.pause(1) 
        plt.close()  
    else:
        plt.legend()
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
    # store = []
    ppg_len = len(ppg)

    for peak in sys_peaks:
        idx = peak
        while idx >= 1 and ppg[idx] >= ppg[idx-1]:
            idx -= 1

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
    ppi = np.diff(ppg[sys_peaks])
    ic(ppi)
    hrv = np.std(ppi)
    return hrv

def find_amplitude_ratios(ppg,fs):
    sys_peaks = find_sys_peaks(ppg,fs)
    amplitude_ratios = ppg[sys_peaks] / np.roll(ppg[sys_peaks], 1)
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
    # ic(sys_peaks)
    # ic(foot_peaks)

    # plot_scatter(ppg,sys_peaks)
    alpha = 0
    for i in range(len(sys_peaks)-1):
        alpha += (sys_peaks[i] - foot_peaks[i] - foot_peaks[i+1])
    alpha /= len(sys_peaks)
    return alpha

def trapezoidal_area(ppg, start_index, end_index, sampling_interval):
    
    segment = ppg[start_index:end_index+1]
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
        area1 = trapezoidal_area(ppg, s1_index, s2_index, sampling_interval)
        area2 = trapezoidal_area(ppg, s3_index, s4_index, sampling_interval)
        ipa_ratio = area1 / area2 if area2 != 0 else np.inf
        if ipa_ratio:
            AREA += ipa_ratio
            total += 1

    return AREA

def find_systolic_time_x(ppg, fs,val = 20):
    # ppg = ppg[::-1]
    sys_peaks = find_sys_peaks(ppg, fs)

    peak_len = len(sys_peaks)
    if peak_len == 0:
        return 0
    
    val += 15
    mul_fact = val/100

    foot = []
    sys_time = 0
    for peak in sys_peaks:
        idx = peak
        while idx >= 1 and ppg[idx] >= ppg[idx - 1]:
            idx -= 1
        add = math.ceil(idx + mul_fact * (peak - idx))
        # print(f"peak: {ppg[peak]}, 10%: {ppg[add]}")
        foot.append(add)
        if peak != add:
            sys_time += ((ppg[peak] - ppg[add])/fs)

    sys_time /= peak_len
    # ic(sys_time)
    return sys_time

def find_pwv(ppg,fs):
    # ppg = ppg[::-1]
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


def extract_features(ppg, fs=26):

    
    ppg = ppg[::-1]
    ppg_filtered = bandpass_filter(ppg)
    sys_peaks = find_sys_peaks(ppg_filtered, fs)

    hr = find_heart_rate(sys_peaks, fs)
    ref_ind = find_refliction_index(ppg_filtered, sys_peaks)
    lasi = find_lasi(ppg, fs)
    crest_time = find_crest_time(ppg_filtered, fs)
    mnpv = find_mnpv(ppg_filtered, fs)
    sys_time, foot_time = find_both_times(ppg_filtered, fs)
    pir = find_pir(ppg_filtered, fs)
    aix = find_augmentation_index(ppg_filtered, fs)
    pulse_height = find_pulse_height(ppg_filtered, fs)
    pulse_width = find_pulse_width(ppg_filtered, fs)
    hrv = find_hrv(ppg_filtered,fs)
    amplitude_ratios = find_amplitude_ratios(ppg_filtered,fs)
    max_amplitude, min_amplitude = find_max_min_amplitudes(ppg_filtered)
    womersley_number = find_womersley_number(ppg_filtered,fs)
    alpha = find_alpha(ppg_filtered,fs)
    ipa = find_ipa(ppg_filtered, fs)
    sys_time_ten = find_systolic_time_x(ppg_filtered, fs)
    pwv = find_pwv(ppg_filtered,fs)


    ret_dict = {
        "alpha":alpha,
        "amplitude_ratios": amplitude_ratios,
        "augmentation_index": aix,
        "crest_time": crest_time,
        "foot_time": foot_time,
        "hr": hr,
        "hrv": hrv,
        "ipa":ipa,
        "lasi": lasi,
        "max_amplitude":max_amplitude,
        "min_amplitude":min_amplitude,
        "mnpv": mnpv,
        "pir": pir,
        "pulse_height": pulse_height,
        "pulse_width": pulse_width,
        "pwv":pwv,
        "ref_ind": ref_ind,
        "sys_time": sys_time,
        "sys_time_ten":sys_time_ten,
        "womersley_number":womersley_number
    }
    return ret_dict

def bandpass_filter(ppg_signal, lowcut=0.5, highcut=5.0, fs=26, order=2):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, ppg_signal)
    return y

def plot_scatter(data,peaks):
    # print("hey bro")
    plt.plot(data)
    plt.scatter(peaks, [data[i] for i in peaks], c='red')
    plt.show()


def print_well(data):
    print(f"\n size of the array is: {len(data)}")
    for i in range(len(data)):
        if i == len(data)-1:
            print(data[i])
        else:
            print(data[i],end = ",")


def plot_single(data):
    data_new = bandpass_filter(data)
    plt.plot(data_new)
    plt.show()

def finder():
    plot_stuff = 0
    if plot_stuff:
        data = [0.364462,2.696179,5.647935,8.045166,8.547613,7.547419,6.214999,5.177676,4.554520,4.165003,3.755753,3.211612,2.408299,1.299919,0.277688,-0.225727,-0.451936,-0.944561,-1.741506,-2.366231,-2.336217,-1.264171,1.109232,4.042153,5.898520,5.963606,4.944267,3.764640,2.818644,2.051447,1.299033,0.470451,-0.434846,-1.417806,-2.521924,-3.567810,-4.270494,-4.703189,-5.058818,-5.265712,-5.275210,-5.192030,-4.618080,-2.568816,0.898603,3.956663,5.138858,4.784944,3.823365,2.794943,2.050570,1.734730,1.646213,1.520812,1.299853,0.975268,0.449211,-0.328550,-1.256872,-2.224370,-3.186678,-4.010086,-4.478682,-4.113347,-2.199187,0.972610,3.666074,4.490150,3.732955,2.597074,1.886467,1.574129,1.258428,0.627402,-0.404150,-1.560720,-2.417927,-2.863816,-3.178167,-3.742051,-4.593915,-5.377263,-5.838221,-5.978482,-5.435896,-3.326316,0.294180,3.605690,4.958875,4.518229,3.290957,1.994065,1.093726,0.744320,0.636900,0.293996,-0.693662,-2.394475,-4.212826,-5.413206,-5.892179,-6.039690,-6.230985,-6.576437,-6.977137,-7.246959,-6.833763,-4.797761,-0.984660,3.025012,5.268659,5.392072,4.372777,3.129491,2.062836,1.242160,0.653607,0.067226,-0.839666,-1.813171,-2.459696,-2.876342,-3.095221,-3.101438,-3.297184,-3.928569,-4.544403,-4.596698,-3.841322,-1.824812,1.578272,5.068219,7.046887,7.408408,7.084656,6.617182,5.999933,5.375520,5.050621,4.934516,4.552560,3.698153,2.734126,2.092354,1.852645,1.758390,1.406818,0.867466,0.767018,1.306698,0.208791,0.043385,0.208791,0.608974,-0.163601,-0.814103,8.547613,-7.246959,-4.225400,3.369377]
        peaks = [4,25,46,67,88,111,133]
        plot_scatter(data,peaks)

    else:
        df = pd.read_csv("valid_overall_new.csv")
        big_array = df.values
        print(len(big_array))
        index = 350
        for i in range(len(big_array)):
            bp = int(big_array[i][-1])
            if bp >= 130 and bp <= 175:
                ic(bp)
                ppg = big_array[i][0:150]
                print_well(ppg)
                print("\n")

            plot_single(ppg)
        feat = extract_features(ppg)
        ic(feat)



def ravi_ampd(ppg):
    ppg_filt = bandpass_filter(ppg)

    # for ele in ppg_filt:
    #     print(ele,end = ",")

    # plt.plot(ppg_filt)
    # plt.show()
    n = len(ppg_filt)
    keep = {}
    max_len = 0
    global_store = []
    for window_len in range(3, n-3):
        low = 0
        high = low + window_len
        store = []

        while high < n:
            for j in range(low, high):
                if j == 0 or j == high-1:
                    continue
                
                if ppg_filt[j] > ppg_filt[j-1] and ppg_filt[j] > ppg_filt[j+1]:
                    already_stored = 0
                    for peak in store:
                        if peak == j:
                            already_stored = 1
                            break
                        
                    if not already_stored:
                        store.append(j)
            low += 1
            high += 1

        store_len = len(store)
        if store_len > max_len:
            max_len = store_len
            global_store.clear()
            global_store = store
        # print(window_len,store)

    print(max_len, global_store)
    plot_scatter(ppg_filt, global_store)

import numpy as np
def ampd(ppg_filt):
    """Apply the AMPD algorithm to find the peaks in the PPG signal."""
    # for ele in ppg_filt:
    #     print(ele,end = ",")
    # print("\n")
    n = len(ppg_filt)
    max_len = 0
    global_store = []

    for window_len in range(3, n - 3):
        low = 0
        high = low + window_len
        store = []

        while high < n:
            max_index = np.argmax(ppg_filt[low:high]) + low
            if max_index not in store and max_index >= 1 and max_index + 1 < n and ppg_filt[max_index] > ppg_filt[max_index - 1] and ppg_filt[max_index] > ppg_filt[max_index + 1]:
                store.append(max_index)
            low += 1
            high += 1

        store_len = len(store)
        if store_len > max_len:
            max_len = store_len
            global_store = store.copy()
    improved_peaks = []
    foot = []


    for peak in global_store:
        idx = peak
        while idx+1 < n and ppg_filt[idx] < ppg_filt[idx+1]:
            idx = idx + 1
        improved_peaks.append(idx)
        while idx+1 < n and ppg_filt[idx] > ppg_filt[idx+1]:
            idx = idx + 1
        foot.append(idx)

    final_peaks = []
    final_peaks.append(improved_peaks[0])
    distance_peaks = 10
    for i in range(len(improved_peaks)):
        if improved_peaks[i] - final_peaks[-1] >= distance_peaks:
            final_peaks.append(improved_peaks[i])
    return final_peaks,foot

def pretty_print(arr,name = None):
    if name is not None:
        print(f"{name}: ",end="")
    print("[",end = "")
    n = len(arr)
    for i in range(n):
        if i == n-1:
            print(arr[i], end="")
        else:
            print(arr[i], end=",")
    
    print("]")
    print("\n")

def after(ppg,sys_peaks,foot):
    ic(sys_peaks)
    ic(foot)
    # plot_scatter(ppg,sys_peaks + foot)
    n = len(foot)
    orig = [ele for ele in ppg]

    for i in range(n - 1):
        min1_idx = foot[i]
        min2_idx = foot[i+1]


        pulse = ppg[min1_idx:min2_idx+1]
        min_val = ppg[min1_idx]
        max_val = np.max(pulse)
        normalized_pulse = (pulse - min_val) / (max_val - min_val)

        for j in range(min1_idx, min2_idx + 1):
            ppg[j] = normalized_pulse[j - min1_idx]

    length = len(ppg)
    cnt = 0
    for i in range(length):
        cnt += ppg[i] == orig[i]
    ic(length,cnt)

    plt.plot(orig,label = "Original")
    plt.plot(ppg, label='Normalised')

    # Draw straight lines at the detected peaks
    peaks = sys_peaks
    for peak in foot:
        peaks.append(peak)
    
    for peak in peaks:
        plt.axvline(x=peak, color='r', linestyle='--', label='Peak' if peak == peaks[0] else "")

    # Highlight the peaks on the plot
    plt.plot(peaks, ppg[peaks], 'ro')

    plt.title('PPG Signal with Detected Peaks')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

    # plt.plot(orig)
    # plt.title("original")
    
    # plt.plot(ppg)
    # plt.title("pulse normalised")
    # plt.show()

def poly_fit_max_slopes(y,start,end):
    x = [i+1 for i in range(len(y))]
    p = Polynomial.fit(x, y, 5)
    x_fit = np.linspace(x[0], x[-1], end - start + 1)
    y_fit = p(x_fit)

    p_deriv = p.deriv()
    y_deriv = p_deriv(x_fit)

    max_slope_index = np.argmax(y_deriv) + start

    # two = 2
    # if start - max_slope_index <= two:
    #     if start + two + 1 < start:
    #         max_slope_index = start + two + 1
    #     else:
    #         max_slope_index = start + two
    
    # ic(start,end,max_slope_index)
    return max_slope_index

def find_max_slopes(ppg_filt,sys_peaks, foot):

    peak_len = len(sys_peaks)
    max_slopes = [sys_peaks[0] - 3]

    for i in range(0,peak_len-1):
        start = foot[i]
        end = sys_peaks[i+1]
        segment = ppg_filt[start:end]
        max_slope_index = poly_fit_max_slopes(segment,start,end)
        # ic(max_slope_index,start,end)
        max_slopes.append(max_slope_index)

    return max_slopes


def fit_poly_diastolic(y, start, end):
    x = [i+1 for i in range(len(y))]
    
    order = 5
    p = Polynomial.fit(x, y, order)
    
    x_fit = np.linspace(x[0], x[-1], end - start + 1)
    y_fit = p(x_fit)
    
    p_deriv1 = p.deriv(1)
    p_deriv2 = p.deriv(2)
    
    y_deriv1 = p_deriv1(x_fit)
    y_deriv2 = p_deriv2(x_fit)
    
    zero_crossings = np.where(np.isclose(y_deriv1, 0, atol=1e-5))[0]
    
    diastolic_indices = [i for i in zero_crossings if y_deriv2[i] < 0]
    diastolic_peak_idx = -1

    if diastolic_indices:
        diastolic_peak_idx = diastolic_indices[0]
    else:
        local_minima_indices = argrelextrema(y_deriv2, np.less)[0]
        if local_minima_indices.size > 0:
            diastolic_peak_idx = local_minima_indices[np.argmin(y_deriv2[local_minima_indices])]
        else:
            diastolic_peak_idx = np.argmin(y_deriv2)
    
    diastolic_peak_x = x_fit[diastolic_peak_idx]
    diastolic_peak_index = int(round(diastolic_peak_x)) + start


    two = 2
    if abs(diastolic_peak_index - start) <= two and diastolic_peak_index + two - 1 < end:
        diastolic_peak_index += two - 1
    
    # ic(start,end, diastolic_peak_index)    
    return diastolic_peak_index


def find_dia_peaks(ppg_filt):
    sys_peaks, foot = ampd(ppg_filt)
    peak_len = len(foot)
    diastolic_peaks = []
    for i in range(peak_len):
        start = sys_peaks[i]
        end = foot[i]
        ascending_segment = ppg_filt[start:end]
        d_p = fit_poly_diastolic(ascending_segment, start, end)
        diastolic_peaks.append(d_p)
    return diastolic_peaks


def find_dicrotic_notch(ppg,sys_peaks,dia_peaks):
    second_derivative = np.gradient(np.gradient(ppg))
    local_maxima_indices = argrelextrema(second_derivative,np.greater)[0]
    valid_maxima = []

    loop_counter = 0
    local_len = len(local_maxima_indices)
    min_length = min(len(sys_peaks),len(dia_peaks))

    for i in range(min_length):
        d_p = dia_peaks[i]
        s_p = sys_peaks[i]
        prev = -1
        while loop_counter < local_len and local_maxima_indices[loop_counter] > s_p and local_maxima_indices[loop_counter] < d_p:
            prev = local_maxima_indices[loop_counter]
            loop_counter += 1

        if prev != -1:
            valid_maxima.append(prev)
        else:
            add = (d_p + s_p)//2
            valid_maxima.append(add)

        
    # ic(valid_maxima)
    return valid_maxima


def plot_scatter_multi(ppg_filt,max_slopes,sys_peaks,dic_notch,dia_peaks,foot):


    plt.plot(ppg_filt, label='PPG Signal')
    plt.scatter(max_slopes, [ppg_filt[i] for i in max_slopes], c='green', label='Max Slopes')
    plt.scatter(sys_peaks, [ppg_filt[i] for i in sys_peaks], c='red', label='Systolic Peaks')
    plt.scatter(dic_notch, [ppg_filt[i] for i in dic_notch], c='blue', label='Dicrotic Notch')
    plt.scatter(dia_peaks, [ppg_filt[i] for i in dia_peaks], c='black', label='Diastolic Peaks')
    plt.scatter(foot, [ppg_filt[i] for i in foot], c='orange', label='Foot')


    plt.legend()
    plt.show()

def main():
    ppg = [116,117,121,123,124,126,127,128,130,131,124,120,121,123,126,127,128,130,131,131,134,135,133,123,123,124,127,128,131,133,134,134,135,137,134,124,123,124,126,127,128,130,133,134,134,134,135,126,123,123,126,128,128,130,131,134,135,135,135,128,126,126,127,128,130,133,133,134,135,135,137,133,124,124,126,128,128,131,133,134,135,135,137,133,124,123,126,128,128,131,131,134,135,135,135,134,124,123,124,126,127,128,130,131,133,135,137,135,126,123,124,126,127,130,133,134,134,135,138,138,133,124,124,127,128,128,133,134,135,135,137,138,135,126,124,127,128,130,131,134,135,137,137,138,138,127,124,124,126,128]
    ppg = np.array(ppg)
    ppg_filt = bandpass_filter(ppg)
    sys_peaks, foot = ampd(ppg_filt)
    max_slopes = find_max_slopes(ppg_filt, sys_peaks, foot)
    dia_peaks = find_dia_peaks(ppg_filt)   
    dic_notch = find_dicrotic_notch(ppg_filt,sys_peaks,dia_peaks)
    ic(max_slopes)
    ic(sys_peaks)
    ic(dic_notch)
    ic(dia_peaks)
    ic(foot)
    # plot_scatter(ppg_filt,max_slopes)
    plot_scatter_multi(ppg_filt,max_slopes,sys_peaks,dic_notch,dia_peaks,foot)
    

    

main()












# ret_dict = extract_features(ppg)
# feature_names = list(ret_dict.keys())
# feature_values = list(ret_dict.values())
# for i in range(len(feature_values)):
#     if i in selected_features:
#         print(feature_values[i],end = ",")










