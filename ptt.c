#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#define MAX_PEAKS 1000
#define nl printf("\n")
static void filter(const double b[5], const double a[5], const double x[174], const double zi[4], double y[174]);
static void filtfilt(const double x_in[150], double y_out[150]);

static void filter(const double b[5], const double a[5], const double x[174], const double zi[4], double y[174])
{
    int k;
    int naxpy;
    int j;
    double as;
    for (k = 0; k < 4; k++)
    {
        y[k] = zi[k];
    }

    memset(&y[4], 0, 170U * sizeof(double));
    for (k = 0; k < 174; k++)
    {
        naxpy = 174 - k;
        if (!(naxpy < 5))
        {
            naxpy = 5;
        }

        for (j = 0; j + 1 <= naxpy; j++)
        {
            y[k + j] += x[k] * b[j];
        }

        naxpy = 173 - k;
        if (!(naxpy < 4))
        {
            naxpy = 4;
        }

        as = -y[k];
        for (j = 1; j <= naxpy; j++)
        {
            y[k + j] += as * a[j];
        }
    }
}

static void filtfilt(const double x_in[150], double y_out[150])
{
    double xtmp;
    double d1;
    int i;
    double y[174];
    double b_y[174];
    double a[4];
    static const double b_a[4] = {-0.19377359529033994, -0.1937735952902776,
                                  0.19377359529029725, 0.19377359529031876};

    static const double dv1[5] = {0.19377359529031332, 0.0, -0.38754719058062664,
                                  0.0, 0.19377359529031332};

    static const double dv2[5] = {1.0, -2.342095632660552, 1.9477115412319015,
                                  -0.80784671655553675, 0.20433295072565855};

    xtmp = 2.0 * x_in[0];
    d1 = 2.0 * x_in[149];
    for (i = 0; i < 12; i++)
    {
        y[i] = xtmp - x_in[12 - i];
    }

    memcpy(&y[12], &x_in[0], 150U * sizeof(double));
    for (i = 0; i < 12; i++)
    {
        y[i + 162] = d1 - x_in[148 - i];
    }

    for (i = 0; i < 4; i++)
    {
        a[i] = b_a[i] * y[0];
    }

    memcpy(&b_y[0], &y[0], 174U * sizeof(double));
    filter(dv1, dv2, b_y, a, y);
    for (i = 0; i < 87; i++)
    {
        xtmp = y[i];
        y[i] = y[173 - i];
        y[173 - i] = xtmp;
    }

    for (i = 0; i < 4; i++)
    {
        a[i] = b_a[i] * y[0];
    }

    memcpy(&b_y[0], &y[0], 174U * sizeof(double));
    filter(dv1, dv2, b_y, a, y);
    for (i = 0; i < 87; i++)
    {
        xtmp = y[i];
        y[i] = y[173 - i];
        y[173 - i] = xtmp;
    }

    memcpy(&y_out[0], &y[12], 150U * sizeof(double));
}

int find_signal_peaks(double signal[], int signal_len, int peaks[], int is_ecg, double ppg_mean, double ecg_mean)
{

    double filtered_signal[150];
    filtfilt(signal, filtered_signal);

    int num_peaks = 0, amplitude_tolerance = 5;
    int store[500] = {0};
    int store_index = 0;
    int peak_distance = 30;
    int low = 0, high = 0, max_index = -1, turn = 1;
    double max_val = -1;
    if (is_ecg)
    {
        // for (int i = 0; i < signal_len; i++)
        // {
        //     printf("index: %d,value: %f\n", i, signal[i]);
        // }
        // nl;
        // nl;
        // nl;
        // nl;
    }

    for (int i = 0; i < signal_len; i++)
    {
        if (signal[i] < 0)
        {
            high = i;
            break;
        }
    }
    // printf("is_ecg: %d, ppg_mean: %f, ecg_mean: %f\n", is_ecg, ppg_mean, ecg_mean);
    double amp_tol = is_ecg ? ecg_mean : ppg_mean + 0.2;
    while (high < signal_len)
    {
        if (signal[high])
        {
            if (signal[high] >= amp_tol && signal[high] > max_val)
            {
                max_val = signal[high];
                max_index = high;
            }
        }
        else
        {
            // peaks[num_peaks++] = max_index;
            store[store_index++] = max_index;
            max_index = -1;
            max_val = -1;
        }
        high++;
    }
    peaks[num_peaks++] = store[0];
    for (int i = 1; i < store_index; i++)
    {
        if (abs(peaks[num_peaks - 1] - store[i]) >= peak_distance)
        {
            peaks[num_peaks++] = store[i];
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
        amplitude_tolerance = 15;
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

    int final_peaks_len = 0, tol = 10;
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

    double ecg_mean = 0.0, ppg_mean = 0.0;

    // good for find_signal_peaks, IG
    double fs_ecg = 192.0;
    double fs_ppg = 26.0;

    // double fs_ecg = ecg_len / 30;
    // double fs_ppg = ppg_len / 30;
    // printf("ecg fs: %f, ppg fs: %f\n", fs_ecg, fs_ppg);

    double diff_ecg_signal[ecg_len - 1];
    double diff_ppg_signal[ppg_len - 1];

    calculate_derivative(ecg_signal, diff_ecg_signal, ecg_len);
    calculate_derivative(ppg_signal, diff_ppg_signal, ppg_len);

    for (int i = 0; i < ecg_len - 1; i++)
    {
        if (diff_ecg_signal[i] >= 1)
        {
            ecg_mean += diff_ecg_signal[i];
        }
    }
    ecg_mean /= ecg_len;

    for (int i = 0; i < ppg_len - 1; i++)
    {
        if (diff_ppg_signal[i] >= 1)
        {
            ppg_mean += diff_ppg_signal[i];
        }
    }
    ppg_mean /= ppg_len;

    *num_r_peaks = find_signal_peaks(diff_ecg_signal, ecg_len - 1, r_peaks, 1, ppg_mean, ecg_mean);
    *num_systolic_peaks = find_signal_peaks(diff_ppg_signal, ppg_len - 1, systolic_peaks, 0, ppg_mean, ecg_mean);

    // *num_r_peaks = ampd(diff_ecg_signal, ecg_len - 1, r_peaks, 1);
    // *num_systolic_peaks = ampd(diff_ppg_signal, ppg_len - 1, systolic_peaks, 0);

    double sum_ptt = 0.0;
    int valid_pairs = 0;
    int tol = 60;
    int num_class = ecg_len / 100;

    for (int i = 0; i < *num_r_peaks && i < *num_systolic_peaks; i++)
    {
        int r_peak_index = r_peaks[i];
        int systolic_peak_index = systolic_peaks[i];

        // int class_r = (r_peak_index / 100) * 100;
        // int class_s = (systolic_peak_index / 100) * 100;

        // printf("r class: %d, s class: %d\n", class_r, class_s);

        if (r_peak_index >= ecg_len || systolic_peak_index >= ppg_len)
        {
            continue;
        }

        // printf("rpeak: %d, syspeak: %d\n", r_peak_index, systolic_peak_index);

        double time_r_peak = (double)r_peak_index / fs_ecg;
        double time_systolic_peak = (double)systolic_peak_index / fs_ppg;

        double ptt = fabs((double)(r_peak_index - systolic_peak_index)) / fs_ppg;
        sum_ptt += ptt;
        valid_pairs++;
    }

    if (valid_pairs == 0)
    {
        printf("No valid peak pairs found.\n");
        return -1;
    }
    double ans = (sum_ptt / valid_pairs);

    // printf("valid_pairs: %d, PTT C: %f\n", valid_pairs, ans);
    // printf("PTT C: %f, ", ans);

    return ans;
}
