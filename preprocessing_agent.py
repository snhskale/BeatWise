import wfdb
import numpy as np
import os
import pandas as pd
from scipy.signal import butter, filtfilt
from biosppy.signals import ecg

# Defining the constants used during our training data preparation
WINDOW_BEFORE = 100
WINDOW_AFTER = 180

def filter_ecg_signal(signal, fs):
    """
    Cleans the ECG signal by applying a standard bandpass filter.
    """
    # Designing the bandpass filter
    nyquist_freq = 0.5 * fs
    low_cutoff = 0.5 / nyquist_freq
    high_cutoff = 40 / nyquist_freq
    b, a = butter(1, [low_cutoff, high_cutoff], btype='band')

    # Applying the filter to the signal
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def detect_r_peaks_biosppy(signal, fs):
    """
    Detects R-peaks using the robust Pan-Tompkins-based algorithm from the biosppy library.
    """
    # The ecg.ecg function from biosppy is a complete processing pipeline.
    # We are interested in the 'rpeaks' key which contains the locations of the R-peaks.
    ecg_data = ecg.ecg(signal=signal, sampling_rate=fs, show=False)
    r_peaks = ecg_data['rpeaks']
    return r_peaks

def preprocess_dat_file(record_path):
    """
    The main function of the agent. Takes a path to a WFDB record (without extension)
    and returns a dictionary of processed data ready for the next agents.
    """
    try:
        # 1. Load the raw signal using the record path
        record = wfdb.rdrecord(record_path)
        signal = record.p_signal[:, 0] # Using the first channel
        fs = record.fs
    except Exception as e:
        print(f"Error loading record {record_path}: {e}")
        return None

    # 2. Filter the signal (for segmentation consistency)
    filtered_signal = filter_ecg_signal(signal, fs)

    # 3. Detect R-peaks in the unfiltered signal using the robust algorithm
    r_peaks = detect_r_peaks_biosppy(signal, fs)

    if r_peaks.size == 0:
        print(f"Warning: No R-peaks detected in record {record_path}.")
        return None

    # 4. Segment the filtered signal around each detected R-peak
    heartbeat_segments = []
    valid_r_peaks = []
    for peak in r_peaks:
        start = peak - WINDOW_BEFORE
        end = peak + WINDOW_AFTER
        if start >= 0 and end < len(filtered_signal):
            segment = filtered_signal[start:end]
            heartbeat_segments.append(segment)
            valid_r_peaks.append(peak)

    if not heartbeat_segments:
        print(f"Warning: Could not extract any valid segments from record {record_path}.")
        return None

    # 5. Format the data for the model
    X = np.array(heartbeat_segments)
    X = np.expand_dims(X, 1)

    # Return a dictionary with all necessary information
    return {
        "signals": X,
        "r_peaks": np.array(valid_r_peaks),
        "fs": fs
    }
