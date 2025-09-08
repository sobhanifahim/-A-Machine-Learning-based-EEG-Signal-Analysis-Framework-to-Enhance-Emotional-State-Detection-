import pandas as pd
import numpy as np
from sklearn.decomposition import FastICA
import pywt
from scipy.signal import firwin, lfilter

# Load EEG
df = pd.read_csv("8 Channel EEG signal data.csv")  # shape: (time_samples, channels)
fs = 256  # sampling frequency (Hz) — update if needed

# FIR Bandpass Filter
def fir_bandpass_filter(signal, lowcut, highcut, fs, numtaps=101):
    """Apply FIR bandpass filter to a signal."""
    # Design filter
    fir_coeff = firwin(numtaps, [lowcut, highcut], pass_zero=False, fs=fs)
    # Apply filter
    filtered_signal = lfilter(fir_coeff, 1.0, signal)
    return filtered_signal

# Apply FIR filter to all channels first
filtered_eeg = np.zeros_like(df.values)
for ch_idx in range(df.shape[1]):
    filtered_eeg[:, ch_idx] = fir_bandpass_filter(df.values[:, ch_idx], 1, 100, fs)

# ICA for artifact removal on FIR-filtered data
ica = FastICA(n_components=filtered_eeg.shape[1], random_state=42)
sources = ica.fit_transform(filtered_eeg)
cleaned_eeg = ica.inverse_transform(sources)

# DWT Band Power Extraction
def dwt_band_power(signal, fs, bands, wavelet='db4', level=6):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    band_powers = {}
    for band_name, (low, high) in bands.items():
        energy = 0
        for scale_idx in range(1, len(coeffs)):
            freq_low = fs / (2 ** (scale_idx + 1))
            freq_high = fs / (2 ** scale_idx)
            if freq_low <= high and freq_high >= low:
                energy += np.sum(np.square(coeffs[scale_idx]))
        band_powers[band_name] = energy / len(signal)
    return band_powers

# Define EEG Bands
bands = {
    "Low_Alpha": (0, 10),
    "High_Alpha": (10, 12),
    "Low_Beta": (13, 20),
    "High_Beta": (20, 30),
    "Low_Gamma": (30, 50),
    "High_Gamma": (50, 180)
}

# Compute Band Powers
all_band_features = []
for ch_idx in range(cleaned_eeg.shape[1]):
    signal = cleaned_eeg[:, ch_idx]
    band_powers = dwt_band_power(signal, fs, bands)
    all_band_features.append(band_powers)

# Save Features
band_df = pd.DataFrame(all_band_features)
band_df.to_csv("EEG_band_power_DWT_FIR_ICA.csv", index=False)
