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
from itertools import product
import os
import random

from tensorflow.keras.optimizers import Adam, SGD # type: ignore
from sklearn.model_selection import GridSearchCV, train_test_split
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
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
    len_sys_peak = len(sys_peaks)
    if len_sys_peak == 0:
        return 0
    
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
    

    s3 = []
    for i in range(len(sys)-1):
        half_pos = int((sys[i] + foot[i+1])/2)
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
    return area



def plot_scatter(data, peaks, close = False):
    plt.plot(data)
    
    plt.scatter(peaks, [data[i] for i in peaks], c='red')
    if close:
        plt.show(block=False)
        plt.pause(2)
        plt.close()
    else:
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
    # plot_scatter(ppg_filtered,sys_peaks,1)

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

    # selected_features = [3, 4, 5, 6, 8, 10, 13, 14, 15, 17]

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

def normalize_data(x):
    normalized = (x-min(x))/(max(x)-min(x))
    return normalized


def normalize_abp(abp, max_bp, min_bp):
    norm_abp = ((abp-min_bp)/(max_bp-min_bp))
    return norm_abp


def denormalize(val, max_bp, min_bp):
    denormalized_bp = (val * (max_bp - min_bp)) + min_bp
    return math.floor(denormalized_bp)





def change_bp(bp):

    if bp >= 140 and bp < 150:
        if bp - 140 < 150 - bp:
            return 140
        else:
            return 150
    elif bp >= 150 and bp < 160:
        if bp - 150 < 160 - bp:
            return 150
        else:
            return 160
    elif bp >= 160 and bp < 170:
        if bp - 160 < 170 - bp:
            return 160
        else:
            return 170
    elif bp >= 170 and bp < 180:
        if bp - 170 < 180 - bp:
            return 170
        else:
            return 180
    elif bp >= 180 and bp < 190:
        if bp - 180 < 190 - bp:
            return 180
        else:
            return 190
    elif bp >= 190 and bp < 200:
        if bp - 190 < 200 - bp:
            return 190
        else:
            return 200
    return bp



def call_knn_regression(X_train_scaled, y_train):
    param_grid = {
        'n_neighbors': range(1, int(len(X_train_scaled)**0.5)+1),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    grid_search = GridSearchCV(
        KNeighborsRegressor(), param_grid, cv=5, scoring='r2')
    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    ic(best_params)


def call_linear_regression(X_train_scaled, y_train):
    linear_reg = LinearRegression()
    linear_reg.fit(X_train_scaled, y_train)
    best_model = linear_reg
    return best_model


def call_ridge_regression(X_train_scaled, y_train):
    ridge_param_grid = {
        'alpha': [0.1, 1.0, 10.0]
    }
    ridge_grid_search = GridSearchCV(
        Ridge(), ridge_param_grid, cv=5, scoring='r2')
    ridge_grid_search.fit(X_train_scaled, y_train)
    best_model = ridge_grid_search.best_estimator_
    return best_model


def call_rf_regression(X_train_scaled, y_train):

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    }
    grid_search = GridSearchCV(
        RandomForestRegressor(), param_grid, cv=5, scoring='r2', verbose=0)
    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_
    return best_model,best_model.get_params


def call_svr_regression(X_train_scaled, y_train):
    param_grid = {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'degree': [2, 3, 4, 5],  # Only used for 'poly' kernel
        'epsilon': [0.1, 0.2, 0.5, 0.3, 0.4, 0.6, 0.7, 0.8, 1],
        'coef0': [0.0, 0.1, 0.5, 1],  # Used for 'poly' and 'sigmoid' kernels
        'shrinking': [True, False],
        'tol': [1e-3, 1e-4, 1e-5],
        'max_iter': [1000, 5000, 10000, -1]  # -1 for no limit
    }

    svr_grid_search = GridSearchCV(
        SVR(), param_grid, cv=2, scoring='r2', verbose=5)
    svr_grid_search.fit(X_train_scaled, y_train)
    best_model = svr_grid_search.best_estimator_
    return best_model,best_model.get_params


def call_gb_regression(X_train_scaled, y_train):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(
        GradientBoostingRegressor(), param_grid, cv=5, scoring='r2', verbose=5)
    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_
    return best_model


def call_ada_boost_regression(X_train_scaled, y_train):
    base_regressor = DecisionTreeRegressor(random_state=42)
    ada_regressor = AdaBoostRegressor(
        estimator=base_regressor, random_state=42)

    param_grid = {
        'estimator__max_depth': [2, 4, 6, 8, 10],
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.5, 1.0]
    }

    grid_search = GridSearchCV(estimator=ada_regressor, param_grid=param_grid,
                               cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)

    best_model = grid_search.best_estimator_
    return best_model,best_model.get_params


def print_sorted_dict_return(bp_dict):
    sorted_dict = {key: bp_dict[key] for key in sorted(bp_dict)}
    for k, v in sorted_dict.items():
        print(k, v)

def write_data(data, single, filename, verbose):
    # file.write("*" * 100 + "\n")

    with open(filename, 'w') as file:
        if single == 1:
            file.write("{")
            for i in range(len(data)):
                item = data[i]
                if i == len(data)-1:
                    file.write(str(item))
                else:
                    file.write(str(item) + ",")
            file.write("}\n")
            if verbose > 0:
                print(f"written in {filename}")
        else:
            file.write("{\n")
            for item in data:
                file.write("{")
                for i in range(len(item)):
                    ele = item[i]
                    if i == len(item)-1:
                        file.write(str(ele))
                    else:
                        file.write(str(ele) + ",")

                file.write("},\n")
            file.write("}")
            if verbose>0:
                print(f"written in {filename}")

def write_well(DATA,filename,single):

    if single == 0:
        with open(filename, 'w') as file:
            file.write("{")
            for data in DATA:
                for i in range(len(data)):
                    item = data[i]
                    if i == len(data)-1:
                        file.write(str(item))
                    else:
                        file.write(str(item) + ",")
                file.write(",\n")
            file.write("}\n")
        print(f"weights written in {filename}")
    else:
        with open(filename, 'w') as file:
            file.write("{")
            for i in range(len(DATA)):
                item = DATA[i]
                if i == len(DATA)-1:
                    file.write(str(item))
                else:
                    file.write(str(item) + ",")
            file.write("}")
        print(f"bias written in {filename}")

        

def create_model(input_dim, num_neurons1, num_neurons2, num_neurons3, activation):
    model = Sequential([
        Dense(num_neurons1, input_dim=input_dim, activation=activation),
        Dense(num_neurons2, activation=activation),
        Dense(num_neurons3, activation=activation),
        Dense(1)
    ])
    return model

def train_nn(features,blood_pressure_data,max_bp,min_bp,selected_features):

    # scaler = StandardScaler()
    # features_scaled = scaler.fit_transform(features)
    # blood_pressure_scaled = (blood_pressure_data - np.mean(blood_pressure_data)) / np.std(blood_pressure_data)

    num_neurons1 = [32, 64]
    num_neurons2 = [16, 32]
    num_neurons3 = [8, 16]

    optimizers = ['adam', 'sgd']
    activations = ['relu', 'tanh']

    batch_sizes = [32, 64]
    epochs_list = [250,500]


    param_grid = list(product(num_neurons1, num_neurons2, num_neurons3, optimizers, activations, batch_sizes, epochs_list))

    features_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    features_scaled = features_scaler.fit_transform(features)
    blood_pressure_scaled = target_scaler.fit_transform(blood_pressure_data.values.reshape(-1, 1)).flatten()

    X_train, X_test, y_train, y_test = train_test_split(features_scaled, blood_pressure_scaled, test_size=0.1, shuffle=True)

    best_loss = float('inf')
    best_params = None
    tuning = False
    nn1 = 32
    nn2 = 16
    nn3 = 8
    optimizer = "adam"
    activation = "relu"
    batch_size = 64
    epochs = int(sys.argv[1])

    if tuning:
        for params in param_grid:
            nn1, nn2, nn3, optimizer, activation, batch_size, epochs = params

            model = create_model(features_scaled.shape[1], nn1, nn2, nn3, activation)
            if optimizer == 'adam':
                opt = Adam()
            elif optimizer == 'sgd':
                opt = SGD()
            
            model.compile(optimizer=opt, loss='mse')
            
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)
            
            loss = model.evaluate(X_test, y_test, verbose=0)

            if loss < best_loss:
                best_loss = loss
                best_params = params
                print(f'Test loss: {loss} for parameters: {params}')


        print(f'Best loss: {best_loss} with parameters: {best_params}')

        nn1, nn2, nn3, optimizer, activation, batch_size, epochs = best_params

    
    model = create_model(features_scaled.shape[1], nn1, nn2, nn3, activation)
    if optimizer == 'adam':
        opt = Adam()
    elif optimizer == 'sgd':
        opt = SGD()

    model.compile(optimizer=opt, loss='mse')

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)

    loss = model.evaluate(X_test, y_test, verbose=1)
    print(f'Best model test loss: {loss}')
    y_pred = model.predict(X_test)

    good = 0
    total = 0
    tol = 20
    save = []
    for actual, pred in zip(y_test, y_pred):
        a = denormalize(actual, max_bp, min_bp)
        b = denormalize(pred, max_bp, min_bp)
        diff = abs(a - b)
        good += (diff <= tol)
        total += 1
        # print(f"Actual : {a}, Predicted: {b}, Difference: {diff}")
        save.append([a,b,diff])

    save.sort(key = lambda x: x[0])

    for a,b,diff in save:
        print(f"Actual : {a}, Predicted: {b}, Difference: {diff}")

    print("\n")
    ic(selected_features)
    total_pass = round((good/total) * 100, 2)
    print(f"Train size: {X_train.shape[0]}, test size: {X_test.shape[0]}")
    print(f"Best test case pass percentage: {total_pass} @ {tol}")

    model.save('nn_regressor_130_175_running.keras')

    print("nn_regressor is saved")
    for i, layer in enumerate(model.layers):
        weights, biases = layer.get_weights()
        write_well(weights, f'model_weights/layer{i+1}_weights.txt', 0)
        write_well(biases, f'model_weights/layer{i+1}_biases.txt', 1)
        
def train_svr(features,blood_pressure_data,max_bp,min_bp):
    
    features_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    features_scaled = features_scaler.fit_transform(features)
    blood_pressure_scaled = target_scaler.fit_transform(blood_pressure_data.values.reshape(-1, 1)).flatten()
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, blood_pressure_scaled, test_size=0.1, shuffle=True)

    param_grid = {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'degree': [2, 3, 4, 5],  
        'epsilon': [0.1, 0.2, 0.5, 0.3, 0.4, 0.6, 0.7, 0.8, 1],
        'coef0': [0.0, 0.1, 0.5, 1], 
        'shrinking': [True, False],
        'tol': [1e-3, 1e-4, 1e-5],
        'max_iter': [1000, 5000, 10000, -1]  
    }

    svr_grid_search = GridSearchCV(SVR(), param_grid, cv=5, scoring='r2', verbose=5)
    svr_grid_search.fit(X_train, y_train)
    best_model = svr_grid_search.best_estimator_
    y_pred = best_model.predict(X_test)


    good = 0
    total = 0
    tol = 20
    save = []
    for actual, pred in zip(y_test, y_pred):
        a = denormalize(actual, max_bp, min_bp)
        b = denormalize(pred, max_bp, min_bp)
        diff = abs(a - b)
        good += (diff <= tol)
        total += 1
        save.append([a,b,diff])

    save.sort(key = lambda x: x[0])

    for a,b,diff in save:
        print(f"Actual : {a}, Predicted: {b}, Difference: {diff}")
    
    total_pass = round((good/total) * 100, 2)
    print(f"Train size: {X_train.shape[0]}, test size: {X_test.shape[0]}")
    print(f"Best test case pass percentage: {total_pass} @ {tol}")


def train_knn(features,blood_pressure_data,max_bp,min_bp):
    features_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    features_scaled = features_scaler.fit_transform(features)
    blood_pressure_scaled = target_scaler.fit_transform(blood_pressure_data.values.reshape(-1, 1)).flatten()
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, blood_pressure_scaled, test_size=0.1, shuffle=True)

    param_grid = {
        'n_neighbors': range(1, int(len(X_train)**0.5)+1),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    grid_search = GridSearchCV(KNeighborsRegressor(), param_grid, cv=5, scoring='r2',verbose=5)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    ic(best_params)
    y_pred = best_model.predict(X_test)

    good = 0
    total = 0
    tol = 20
    save = []
    for actual, pred in zip(y_test, y_pred):
        a = denormalize(actual, max_bp, min_bp)
        b = denormalize(pred, max_bp, min_bp)
        diff = abs(a - b)
        good += (diff <= tol)
        total += 1
        save.append([a,b,diff])

    save.sort(key = lambda x: x[0])

    for a,b,diff in save:
        print(f"Actual : {a}, Predicted: {b}, Difference: {diff}")
    
    total_pass = round((good/total) * 100, 2)
    print(f"Train size: {X_train.shape[0]}, test size: {X_test.shape[0]}")
    print(f"Best test case pass percentage: {total_pass} @ {tol}")

def prepare():
    dataset = []
    abp = []
    bp_dict = {}
    df = pd.read_csv("valid_overall_new.csv", index_col=None)
    DATA = df.values
    ic(len(DATA))

    low = 130
    high = 170


    ###############################################################################
    # working good, giving different values not stuck in same values range(maybe not)
    # low = 130
    # high = 175
    ###############################################################################


    ret_cols = []
    check_bp_cont = {}

    for i in range(len(DATA)):
        bp = int(DATA[i][-1])
        # bp = change_bp(bp)
        if bp >= low and bp <= high:
            check_bp_cont[bp] = check_bp_cont.setdefault(bp,0) + 1


    # print_sorted_dict_return(check_bp_cont)
    # return

    # selected_features = [3, 4, 5, 6, 8, 10, 13, 14, 15, 17]
    selected_features = sorted(random.sample(range(21), 10))
    # ic(selected_features)

    for i in range(len(DATA)):
        ppg = DATA[i]
        bp = int(DATA[i][-1])
        # bp = change_bp(int(DATA[i][-1]))
        ppg = ppg[:-1]
        
        if bp >= low and bp <= high:

            bp_dict[bp] = bp_dict.setdefault(bp,0) + 1

            ret_dict = extract_features(ppg)
            feature_names = list(ret_dict.keys())
            feature_values = list(ret_dict.values())

            if len(ret_cols) == 0:
                ret_cols = [feature_names[j] for j in range(len(feature_names)) if j in selected_features]
               
            fin_feature_values = [feature_values[j] for j in range(len(feature_values)) if j in selected_features]
            dataset.append(fin_feature_values)
            abp.append(bp)

    max_bp = max(abp)
    min_bp = min(abp)
    ic(max_bp)
    ic(min_bp)


    

    df = pd.DataFrame(dataset, columns=ret_cols)
    df["blood_pressure"] = abp

    ic(df.shape)

    X = df.drop(columns=['blood_pressure'])
    y = df['blood_pressure']
    ic(df.shape)

    # features and the target variable are getting scaled in the train nn function
    train_nn(X,y,max_bp,min_bp,selected_features)
    # train_svr(X,y,max_bp,min_bp)
    # train_knn(X,y,max_bp,min_bp)
    return

    index = 0
    max_r2 = 0
    save_y_test = []
    save_y_pred = []
    save_best_scaler = None
    save_best_model = None
    save_best_params = None

    which_model = "stuff"
    what_model = sys.argv[1]
    limit = int(sys.argv[2])
    print(f"Running: {what_model.upper()}, epoch: {limit}")

    X_train_scaled = []
    X_test_scaled = []

    best_model = None
    save_best_params = None
    save_X_test = []
    save_y_test = []
    
    while index < limit:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

        scaler = StandardScaler()
        save_best_scaler = scaler
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        

        if what_model == "knn":
            which_model = "KNN"
            best_model, best_params = call_knn_regression(X_train_scaled, y_train)
            save_best_model = best_model
            save_best_params = best_params

        if what_model == "ridge":
            which_model = "RIDGE"
            best_model = call_ridge_regression(X_train_scaled, y_train)
            save_best_model = best_model

        if what_model == "linear":
            which_model = "LINEAR"
            best_model = call_linear_regression(X_train_scaled, y_train)
            save_best_model = best_model

        if what_model == "svr":
            which_model = "SVR"
            best_model, best_params = call_svr_regression(X_train_scaled, y_train)
            save_best_model = best_model

        if what_model == "rf":
            which_model = "RF"
            best_model,best_params = call_rf_regression(X_train_scaled, y_train)
            save_best_model = best_model
            save_best_params = best_params

        if what_model == "gb":
            which_model = "GB"
            best_model = call_gb_regression(X_train_scaled, y_train)
            save_best_model = best_model

        if what_model == "ada":
            which_model = "ADA"
            best_model,best_params = call_ada_boost_regression(X_train_scaled, y_train)
            save_best_model = best_model
            save_best_params = best_params
        

        y_pred = best_model.predict(X_test_scaled)
        save_y_pred = y_pred

        save_X_train = X_train_scaled
        save_y_train = y_train.values
        save_X_test = X_test_scaled
        save_y_test = y_test.values

        r2 = r2_score(y_test, y_pred)

        if r2 > max_r2:
            max_r2 = r2
            print(f"index: {index + 1}, r2: {max_r2}")
            save_y_test = y_test
            save_y_pred = y_pred
            
            save_best_model = best_model
            save_best_scaler = scaler
            save_best_params = best_params
            if max_r2  > 0.8:
                print(f"Early stopping: {max_r2}\n")
                break
            # best_shape = support_vectors.shape

        print(f"index: {index + 1}, r2: calculating")
        index += 1

    

    good = 0
    total = 0
    for actual, pred in zip(save_y_test, save_y_pred):
        a = denormalize(actual, max_bp, min_bp)
        b = denormalize(pred, max_bp, min_bp)
        diff = abs(a - b)
        good += (diff <= 20)
        total += 1
        print(f"Actual : {a}, Predicted: {b}, Difference: {abs(a - b)}")    
    pass_test_case = (good/len(save_y_test)) * 100
    print(f"passed test case: {round(pass_test_case,2)}")

    turn = 1
    while True:
        file_path = f'saved_model/{which_model}_model_valid_{turn}.joblib'
        if os.path.exists(file_path):
            turn += 1
        else:
            break

    print(f"Saving the best model and scaler")
    joblib.dump(save_best_model,
                f'saved_model/{which_model}_model_valid_{turn}.joblib')
    joblib.dump(save_best_scaler,
                f'saved_model/{which_model}_scaler_valid_{turn}.joblib')

    print(f"\nBest r2 score: {max_r2}")
    ic(max_bp, min_bp, what_model, turn)



    metadata_bp = {}
    metadata_bp["max_bp"] = max_bp
    metadata_bp["min_bp"] = min_bp
    metadata_bp["what_model"] = what_model
    metadata_bp["turn"] = turn
  

    with open("metadata.json", "w") as handle:
        json.dump(metadata_bp, handle)
        print("metadata written successfully")





prepare()

# print_column_labels()
