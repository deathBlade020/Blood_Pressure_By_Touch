#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#define MAX_PEAKS 1000
#define nl printf("\n")

int find_systolic_peaks(double *signal, int signal_len, int peaks[], int is_ecg)
{

    int num_peaks = 0, amplitude_tolerance = 5;
    int store[500] = {0};
    int store_index = 0;
    if (is_ecg)
    {
        amplitude_tolerance = 130;
    }

    for (int i = 0; i < signal_len; i++)
    {
        if (signal[i] >= amplitude_tolerance)
        {
            store[store_index++] = i;
            // if (!is_ecg)
            // {
            //     printf("peak: %d, value: %f\n", store[store_index - 1], signal[store[store_index - 1]]);
            // }
        }
    }
    int CLASS_SIZE = 100;
    if (!is_ecg)
    {
        CLASS_SIZE = 10;
    }

    int MAX_CLASS = signal_len / CLASS_SIZE;

    double max_val[MAX_CLASS];
    int max_peaks[MAX_CLASS];

    for (int i = 0; i < MAX_CLASS; i++)
    {
        max_val[i] = -1.0;
        max_peaks[i] = -1;
    }

    for (int i = 0; i < store_index; i++)
    {
        int peak = store[i];
        int class = ((peak - CLASS_SIZE) / CLASS_SIZE);
        if (class < 0 || class >= MAX_CLASS)
        {
            continue;
        }
        double amplitude_value = signal[peak];
        if (amplitude_value > max_val[class])
        {
            max_val[class] = amplitude_value;
            max_peaks[class] = peak;
        }
    }
    peaks[num_peaks++] = max_peaks[0];
    int peak_distance = 10;
    for (int i = 1; i < MAX_CLASS; i++)
    {
        if (max_peaks[i] != -1 && abs(peaks[num_peaks - 1] - max_peaks[i]) >= peak_distance)
        {
            int curr_peak = max_peaks[i];
            while (curr_peak + 1 < signal_len && signal[curr_peak] < signal[curr_peak + 1])
            {
                curr_peak++;
            }
            peaks[num_peaks++] = curr_peak;
        }
    }

    return num_peaks;
}

int ampd(double *ppg_filt, int n, int final_peaks[], int is_ecg)
{

    int max_len = 0;
    int global_peaks[MAX_PEAKS] = {0};
    int global_peaks_len = 0;

    for (int window_len = 3; window_len < n - 3; window_len++)
    {
        int store[MAX_PEAKS];
        int store_len = 0;

        for (int low = 0; low + window_len < n; low++)
        {
            int max_index = low;
            double max_ppg_val = ppg_filt[low];

            for (int j = low; j < low + window_len; j++)
            {
                if (ppg_filt[j] >= max_ppg_val)
                {
                    max_ppg_val = ppg_filt[j];
                    max_index = j;
                }
            }

            if (max_index >= 1 && max_index + 1 < n && ppg_filt[max_index] > ppg_filt[max_index - 1] && ppg_filt[max_index] > ppg_filt[max_index + 1])
            {
                int duplicate = 0;
                for (int k = 0; k < store_len; k++)
                {
                    if (store[k] == max_index)
                    {
                        duplicate = 1;
                        break;
                    }
                }

                if (!duplicate)
                {
                    store[store_len] = max_index;
                    store_len++;
                }
            }
        }

        if (store_len > max_len)
        {
            max_len = store_len;
            global_peaks_len = store_len;
            for (int i = 0; i < store_len; i++)
            {
                global_peaks[i] = store[i];
            }
        }
    }

    int improved_peaks[MAX_PEAKS] = {0};
    int improved_peaks_len = 0, amplitude_tolerance = 6;

    if (is_ecg)
    {
        amplitude_tolerance = 100;
    }

    for (int i = 0; i < global_peaks_len; i++)
    {
        int peak = global_peaks[i];
        while (peak + 1 < n && ppg_filt[peak] < ppg_filt[peak + 1])
        {
            peak++;
        }
        if (ppg_filt[peak] >= amplitude_tolerance)
        {
            improved_peaks[improved_peaks_len++] = peak;
        }
    }

    int final_peaks_len = 0, tol = 7;
    final_peaks[final_peaks_len++] = improved_peaks[0];

    for (int i = 1; i < improved_peaks_len; i++)
    {
        if (improved_peaks[i] - final_peaks[final_peaks_len - 1] >= tol)
        {
            final_peaks[final_peaks_len++] = improved_peaks[i];
        }
    }

    return final_peaks_len;
}

void print_peaks(int peaks_array[], int n)
{
    printf("printing peaks\n");
    for (int i = 0; i < n; i++)
    {
        printf("%d,", peaks_array[i]);
    }
    nl;
    nl;
    nl;
    nl;
}

void print_signal(double signal_array[], int n)
{
    printf("printing signal\n");
    for (int i = 0; i < n; i++)
    {
        printf("%f,", signal_array[i]);
    }
    nl;
    nl;
    nl;
    nl;
}

void calculate_derivative(const double signal[], double derivative[], int length)
{
    for (int i = 1; i < length; i++)
    {
        derivative[i - 1] = signal[i] - signal[i - 1];
    }
}

double find_ptt(double ppg_signal_1[], double ppg_signal_2[], int n, int *peak1, int *peak2, int *num_peaks)
{
    // double fs = 200.0;
    double fs = (double)n / 30;
    printf("Sampling frequency: %f\n", fs);

    double diff_ppg_signal_1[n - 1];
    double diff_ppg_signal_2[n - 1];
    calculate_derivative(ppg_signal_1, diff_ppg_signal_1, n);
    calculate_derivative(ppg_signal_2, diff_ppg_signal_2, n);

    int final_peaks1[MAX_PEAKS] = {0};
    int final_peaks2[MAX_PEAKS] = {0};
    int final_peaks_len1 = ampd(diff_ppg_signal_1, n - 1, final_peaks1, 0);
    int final_peaks_len2 = ampd(diff_ppg_signal_2, n - 1, final_peaks2, 0);

    // Determine the minimum length for peak matching
    int min_length = (final_peaks_len1 < final_peaks_len2) ? final_peaks_len1 : final_peaks_len2;
    double average_ptt = 0.0;
    int peak_count = 0;

    int keep1[n - 1];
    int keep2[n - 1];
    for (int i = 0; i < n - 1; i++)
    {
        keep1[i] = 0;
        keep2[i] = 0;
    }
    for (int i = 0; i < final_peaks_len1; i++)
    {
        if (final_peaks1[i] != -1)
        {
            keep1[final_peaks1[i]]++;
        }
    }

    for (int i = 0; i < final_peaks_len2; i++)
    {
        if (final_peaks2[i] != -1)
        {
            keep2[final_peaks2[i]]++;
        }
    }

    int start = 0, end = n - 2, tol = 5;

    for (int i = tol; i < n - 1; i++)
    {
        if (keep1[i] && keep2[i])
        {
            start = i;
            break;
        }
    }

    for (int i = start; i < n - 1; i++)
    {
        if (keep1[i] && keep2[i] && i < min_length - tol)
        {
            end = i;
        }
    }
    printf("start: %d, end: %d\n", start, end);
    int right_shift_tol = 0;

    for (int i = 0; i < min_length; i++)
    {
        int peak1_val = final_peaks1[i], peak2_val = final_peaks2[i];

        if (peak2_val != -1)
        {
            peak2_val += right_shift_tol;
        }

        if (peak1_val != -1 && peak2_val != -1 && peak1_val != peak2_val)
        {
            peak1[peak_count] = peak1_val;
            peak2[peak_count] = peak2_val - right_shift_tol;

            double peak_diff = fabs((double)(peak1_val - peak2_val)) / fs;
            // printf("peak1 = %d, peak2 = %d, peak_diff = %f\n", peak1_val, peak2_val, peak_diff);
            average_ptt += peak_diff;
            peak_count++;
        }
    }

    if (peak_count > 0)
    {
        average_ptt /= peak_count;
        printf("PTT from C: %f\n", average_ptt);
        *num_peaks = peak_count;
        return average_ptt;
    }
    else
    {
        *num_peaks = 0;
        return -1;
    }
}

double calculate_ptt(double ecg_signal[], double ppg_signal[], int ecg_len, int ppg_len, int r_peaks[], int systolic_peaks[], int *num_r_peaks, int *num_systolic_peaks)
{
    double fs_ecg = 200.0;
    double fs_ppg = 26.0;

    // double fs_ecg = ecg_len / 30;
    // double fs_ppg = ppg_len / 30;

    double diff_ecg_signal[ecg_len - 1];
    double diff_ppg_signal[ppg_len - 1];

    calculate_derivative(ecg_signal, diff_ecg_signal, ecg_len);
    calculate_derivative(ppg_signal, diff_ppg_signal, ppg_len);

    *num_r_peaks = find_systolic_peaks(diff_ecg_signal, ecg_len - 1, r_peaks, 1);
    *num_systolic_peaks = find_systolic_peaks(diff_ppg_signal, ppg_len - 1, systolic_peaks, 0);

    double sum_ptt = 0.0;
    int valid_pairs = 0;
    int tol = 50;
    int num_class = ecg_len / 100;
    double store_value[num_class];

    for (int i = 0; i < *num_r_peaks && i < *num_systolic_peaks; i++)
    {
        int r_peak_index = r_peaks[i];
        int systolic_peak_index = systolic_peaks[i];
        int class_r = (r_peak_index / 100) * 100;
        int class_s = (systolic_peak_index / 100) * 100;
        // printf("r class: %d, s class: %d\n", class_r, class_s);

        if (r_peak_index >= ecg_len || systolic_peak_index >= ppg_len)
        {
            continue;
        }

        // printf("rpeak: %d, syspeak: %d\n", r_peak_index, systolic_peak_index);

        double time_r_peak = (double)r_peak_index / fs_ecg;
        double time_systolic_peak = (double)systolic_peak_index / fs_ppg;

        double ptt = fabs(time_r_peak - time_systolic_peak);
        sum_ptt += ptt;
        valid_pairs++;
    }

    if (valid_pairs == 0)
    {
        printf("No valid peak pairs found.\n");
        return -1;
    }
    double ans = (sum_ptt / valid_pairs);
    printf("valid_pairs: %d, PTT C: %f\n", valid_pairs, ans);

    return ans;
}
