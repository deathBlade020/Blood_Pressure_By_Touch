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
void print_new_lines()
{
    nl;
    nl;
    nl;
    nl;
}

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

int find_signal_peaks(double signal[], int signal_len, int peaks[], int is_ecg, int to_print)
{

    double ecg_mean = 0;
    int div = 0;
    int store[MAX_PEAKS];
    int store_index = 0;
    int num_peaks = 0, amplitude_tolerance = 5, peak_distance = 10;
    if (is_ecg)
    {
        if (to_print)
        {
            printf("ecg printing\n");
            for (int i = 0; i < signal_len; i++)
            {
                printf("%f,", signal[i]);
            }
            print_new_lines();
        }
        for (int i = 0; i < signal_len; i++)
        {
            if (signal[i] >= 1)
            {
                ecg_mean += signal[i];
                div++;
            }
        }

        ecg_mean /= div;
        // printf("ecg mean: %f\n", ecg_mean);
        // int window_size = 40;
        int window_size = 50, skip_right = 6, end_tol = 15;
        for (int i = 0; i < signal_len;)
        {
            int start = i, end = i + window_size;
            if (start >= signal_len || end >= signal_len)
            {
                break;
            }
            int max_index = -1;
            double max_val = -1.0;
            for (int j = start; j < end; j++)
            {
                if (signal[j] > max_val)
                {
                    max_val = signal[j];
                    max_index = j;
                }
            }
            if (max_index != -1)
            {

                int q_peak = max_index;
                while (q_peak >= 1 && signal[q_peak] >= signal[q_peak - 1])
                {
                    q_peak--;
                }
                int s_peak = max_index;
                // this will find the s peak
                while (s_peak + 1 < end + end_tol && signal[s_peak] >= signal[s_peak + 1])
                {
                    s_peak++;
                }
                double r_to_q = (signal[max_index] - signal[q_peak]);
                double r_to_s = (signal[max_index] - signal[s_peak]);

                if (max_index >= 1 && max_index + 1 < signal_len && signal[max_index - 1] < signal[max_index] && signal[max_index] > signal[max_index + 1] && r_to_q >= 40 && r_to_s >= 45)
                {
                    store[store_index++] = max_index; // rpeak here
                    // printf("rpeak: %d\n", max_index);
                    // printf("rpeak: %d, diff1: %f, diff2: %f\n", max_index, r_to_q, r_to_s);
                }

                // printf("qpeak: %f, rpeak: %f, speak: %f\n", signal[q_peak], signal[max_index], signal[s_peak]);
                i = s_peak + skip_right;
            }
            else
            {
                i++;
            }

            // peaks[num_peaks++] = max_index;
        }

        peaks[num_peaks++] = store[0];
        for (int i = 1; i < store_index; i++)
        {
            if (store[i] - peaks[num_peaks - 1] >= 30)
            {
                int pp = store[i];
                while (pp + 1 < signal_len && signal[pp] <= signal[pp + 1])
                {
                    pp++;
                }
                peaks[num_peaks++] = pp;
            }
        }
    }
    else
    {
        for (int i = 0; i < signal_len; i++)
        {
            int peak_index = i;
            while (peak_index + 1 < signal_len && signal[peak_index] >= signal[peak_index + 1])
            {
                peak_index++;
            }
            int left = 4, right = 6;
            // printf("peak diff: %d\n", peak_index - i);
            if (peak_index - left >= 0 && signal[peak_index - left] > signal[peak_index] && peak_index + right < signal_len && signal[peak_index] > signal[peak_index + right])
            {
                peak_index += right;
                while (peak_index + 1 < signal_len && signal[peak_index] >= signal[peak_index + 1])
                {
                    peak_index++;
                }
            }

            // valid foot checking
            if (signal[peak_index - left] > signal[peak_index] && signal[peak_index] < signal[peak_index + left])
            {
                store[store_index++] = peak_index; // foot here
                peaks[num_peaks++] = peak_index;
            }
            else
            {
                peak_index = i + 2;
                continue;
            }

            while (peak_index + 1 < signal_len && signal[peak_index] <= signal[peak_index + 1])
            {
                peak_index++;
            }
            i = peak_index;
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
        amplitude_tolerance = 24;
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

    int final_peaks_len = 0, tol = 15;
    final_peaks[final_peaks_len++] = improved_peaks[0];

    for (int i = 1; i < improved_peaks_len; i++)
    {
        if (improved_peaks[i] - final_peaks[final_peaks_len - 1] >= tol)
        {
            final_peaks[final_peaks_len++] = improved_peaks[i];
        }
    }
    if (is_ecg)
    {
        for (int i = 0; i < final_peaks_len; i++)
        {
            printf("peak: %d, value %f\n", final_peaks[i], ppg_filt[final_peaks[i]]);
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

void filter_here(double signal[], int signal_len, double mother_signal[], int is_ecg)
{
    int mother_index = 0;
    for (int i = 0; i < signal_len; i += 150)
    {
        int start = i, end = i + 150;
        if (end > signal_len)
        {
            continue;
        }
        // printf("is_ecg: %d, start: %d, end: %d\n", is_ecg, i, i + 150);

        double temp_signal[150];
        int idx = 0;
        for (int j = start; j < end; j++)
        {
            temp_signal[idx] = 0.0;
            temp_signal[idx++] = signal[j];
        }

        double filtered_signal[150];
        filtfilt(temp_signal, filtered_signal);
        for (int k = 0; k < 150; k++)
        {
            mother_signal[mother_index++] = filtered_signal[k];
        }
    }
}

int max_peak(int a, int b)
{
    return a > b ? a : b;
}
int min_peak(int a, int b)
{
    return a < b ? a : b;
}

double calculate_ptt(double ecg_signal[], double ppg_signal[], int ecg_len, int ppg_len, int r_peaks[], int systolic_peaks[], int *num_r_peaks, int *num_systolic_peaks, double ecg_filt[], double ppg_filt[], int bp)
{

    for (int i = 0; i < ecg_len; i++)
    {
        ecg_filt[i] = ecg_signal[i];
    }

    for (int i = 0; i < ppg_len; i++)
    {
        ppg_filt[i] = ppg_signal[i];
    }

    double fs_ecg = 200.0;
    double fs_ppg = 25.0;

    int is_ecg = 1, to_print = 0;
    *num_r_peaks = find_signal_peaks(ecg_signal, ecg_len, r_peaks, is_ecg, to_print);
    *num_systolic_peaks = find_signal_peaks(ppg_signal, ppg_len, systolic_peaks, 1 - is_ecg, to_print);

    double sum_ptt = 0.0;
    int valid_pairs = 0, num_pulse = 5;

    for (int i = 0; i < num_pulse; i++)
    {
        int r_peak_index = r_peaks[i];
        int systolic_peak_index = systolic_peaks[i];

        if (r_peak_index >= ecg_len || systolic_peak_index >= ppg_len)
        {
            continue;
        }

        // printf("rpeak: %d, syspeak: %d\n", r_peak_index, systolic_peak_index);

        // double time_r_peak = (double)r_peak_index / fs_ecg;
        // double time_systolic_peak = (double)systolic_peak_index / fs_ppg;

        double ptt = (double)(fabs(r_peak_index - systolic_peak_index)) / (fs_ecg * fs_ppg);
        ptt *= fs_ppg;
        sum_ptt += ptt;
        valid_pairs++;
    }
    double consecutive_diff = 0.0;
    int store[num_pulse];
    for (int i = 0; i < num_pulse; i++)
    {
        int rp = r_peaks[i], sp = systolic_peaks[i];
        store[i] = 0;
        store[i] = max_peak(rp, sp) - min_peak(rp, sp);
    }
    
    int diff_30 = 0, allowed_limit = 30;
    for (int i = 1; i < num_pulse; i++)
    {
        diff_30 += (store[i] - store[i - 1] <= allowed_limit);
    }

    int consecutive_diff_result = (diff_30 >= num_pulse - 2);
    double ptt_ret = (sum_ptt / valid_pairs);

    // double equation = consecutive_diff_result + (1 / ptt_ret);

    int ppg_is_high = (consecutive_diff_result && ptt_ret >= 0.3);
    
    printf("BP: %d, consective_diff: %d, PTT: %.2f, ppg_is_high: %d\n", bp, consecutive_diff_result, ptt_ret, ppg_is_high);

    if (valid_pairs == 0)
    {
        printf("No valid peak pairs found.\n");
        return -1;
    }

    return ptt_ret;
}
