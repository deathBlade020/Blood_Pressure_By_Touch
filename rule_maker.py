import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.optimize import minimize

# Sample data
bp = [98, 153, 128, 157, 106, 100, 135, 125, 140, 104, 108, 102, 105, 104, 101, 110, 131, 90, 100, 150, 117, 96, 131, 112, 130, 89, 128, 106, 188, 117, 117, 115, 98, 102, 147, 119, 119, 120, 124, 120, 130, 108, 110, 110, 174,
      101, 140, 95, 150, 115, 93, 114, 103, 138, 104, 145, 101, 119, 167, 110, 152, 84, 104, 166, 169, 146, 105, 113, 153, 99, 135, 114, 106, 112, 128, 112, 150, 133, 92, 117, 122, 132, 141, 125, 116, 124, 113, 102, 97, 169, 99]
pulse_amplitude = [15.600000, 13.400000, 8.200000, 28.200000, 23.800000, 22.400000, 41.600000, 8.000000, 4.200000, 8.200000, 27.400000, 7.200000, 12.600000, 8.600000, 26.600000, 16.800000, 29.000000, 4.800000, 11.000000, 29.200000, 35.400000, 33.600000, 9.000000, 20.200000, 21.400000, 8.400000, 11.200000, 8.400000, 36.600000, 22.600000, 18.000000, 15.400000, 32.800000, 34.600000, 30.400000, 30.600000, 23.800000, 16.200000, 24.200000, 18.200000, 18.000000, 12.400000, 37.800000, 20.200000, 20.800000,
                   4.800000, 19.600000, 16.600000, 11.800000, 17.000000, 28.400000, 22.200000, 14.000000, 15.200000, 9.600000, 10.800000, 16.000000, 10.800000, 36.200000, 11.000000, 26.000000, 5.600000, 3.200000, 20.600000, 7.600000, 20.400000, 7.600000, 13.600000, 8.200000, 25.400000, 14.800000, 7.200000, 6.600000, 12.800000, 8.200000, 14.600000, 18.000000, 19.400000, 28.200000, 11.600000, 19.200000, 24.200000, 8.400000, 25.000000, 20.000000, 10.400000, 9.800000, 33.200000, 26.400000, 21.200000, 5.400000]
pulse_widths = [8.040000, 9.720000, 13.600000, 8.680000, 9.880000, 9.120000, 11.480000, 12.920000, 8.280000, 10.360000, 8.440000, 6.880000, 7.080000, 8.560000, 11.640000, 7.960000, 7.320000, 12.040000, 11.720000, 12.480000, 8.720000, 11.560000, 9.760000, 6.880000, 7.920000, 8.960000, 11.680000, 7.960000, 9.440000, 8.200000, 15.080000, 8.080000, 8.400000, 9.760000, 10.040000, 9.240000, 13.960000, 9.160000, 5.880000, 9.840000, 9.960000, 10.400000, 10.640000, 6.800000, 7.960000,
                7.000000, 8.200000, 9.880000, 6.480000, 12.200000, 8.800000, 14.800000, 8.480000, 7.080000, 7.240000, 12.040000, 8.280000, 12.320000, 11.200000, 7.920000, 8.840000, 8.880000, 7.440000, 11.560000, 8.560000, 8.640000, 8.400000, 8.960000, 7.120000, 7.680000, 11.240000, 12.520000, 10.000000, 9.800000, 11.360000, 11.720000, 7.160000, 8.440000, 10.800000, 7.480000, 8.440000, 11.480000, 6.880000, 9.600000, 9.960000, 7.440000, 9.760000, 8.920000, 14.600000, 9.320000, 7.400000]


# Define high BP threshold
high_bp_threshold = 140

# Get indices of high BP values
high_bp_indices = [i for i, value in enumerate(
    bp) if value >= high_bp_threshold]

# Extract pulse amplitudes and widths for high BP
high_bp_amplitudes = [pulse_amplitude[i] for i in high_bp_indices]
high_bp_widths = [pulse_widths[i] for i in high_bp_indices]

# Extract pulse amplitudes and widths for low/normal BP
low_normal_bp_indices = [i for i in range(len(bp)) if i not in high_bp_indices]
low_normal_bp_amplitudes = [pulse_amplitude[i] for i in low_normal_bp_indices]
low_normal_bp_widths = [pulse_widths[i] for i in low_normal_bp_indices]

# Define a function to calculate the confusion matrix score


def calculate_score(params):
    left_min, left_max, right_min, right_max = params

    # Prediction for high BP samples
    high_bp_labels = [
        (amplitude >= left_min and amplitude <=
         left_max and width >= right_min and width <= right_max)
        for amplitude, width in zip(high_bp_amplitudes, high_bp_widths)
    ]

    # Prediction for low/normal BP samples
    low_normal_bp_labels = [
        (amplitude >= left_min and amplitude <=
         left_max and width >= right_min and width <= right_max)
        for amplitude, width in zip(low_normal_bp_amplitudes, low_normal_bp_widths)
    ]

    y_true = [1] * len(high_bp_labels) + [0] * len(low_normal_bp_labels)
    y_pred = high_bp_labels + low_normal_bp_labels

    conf_matrix = confusion_matrix(y_true, y_pred)

    # Use true positives for maximizing
    return -conf_matrix[1, 1]  # Negative to maximize true positives


# Initial guesses for the thresholds
initial_params = [10, 30, 18, 20]

# Optimize the thresholds
result = minimize(calculate_score, initial_params, bounds=[(0, 50)]*4)
best_params = result.x
left_min, left_max, right_min, right_max = best_params

print(f"Optimal Parameters: left_min={left_min:.2f}, left_max={
      left_max:.2f}, right_min={right_min:.2f}, right_max={right_max:.2f}")

# Example usage
print("Example rule prediction on high BP samples:")
for amplitude, width in zip(high_bp_amplitudes, high_bp_widths):
    print(f"Amplitude: {amplitude:.2f}, Width: {width:.2f} => High BP: {
          amplitude >= left_min and amplitude <= left_max and width >= right_min and width <= right_max}")

print("\nExample rule prediction on low/normal BP samples:")
for amplitude, width in zip(low_normal_bp_amplitudes, low_normal_bp_widths):
    print(f"Amplitude: {amplitude:.2f}, Width: {width:.2f} => High BP: {
          amplitude >= left_min and amplitude <= left_max and width >= right_min and width <= right_max}")
