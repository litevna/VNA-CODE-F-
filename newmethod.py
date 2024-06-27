import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def read_s2p_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    freq = []
    s11 = []
    s21 = []
    for line in lines:
        if not line.startswith('!') and not line.startswith('#') and line.strip():
            parts = line.split()
            freq.append(float(parts[0]))
            s11.append(complex(float(parts[1]), float(parts[2])))
            s21.append(complex(float(parts[3]), float(parts[4])))

    freq = np.array(freq)
    s11 = np.array(s11)
    s21 = np.array(s21)

    return freq, s11, s21

def combine_s2p_data(folder):
    # Get a list of all .s2p files in the folder
    s2p_files = glob.glob(os.path.join(folder, '*.s2p'))

    combined_freq = []
    combined_s11 = []
    combined_s21 = []

    for s2p_file in s2p_files:
        freq, s11, s21 = read_s2p_file(s2p_file)
        combined_freq.extend(freq)
        combined_s11.extend(s11)
        combined_s21.extend(s21)

    combined_freq = np.array(combined_freq)
    combined_s11 = np.array(combined_s11)
    combined_s21 = np.array(combined_s21)

    # Sort by frequency to ensure data is ordered correctly
    sorted_indices = np.argsort(combined_freq)
    combined_freq = combined_freq[sorted_indices]
    combined_s11 = combined_s11[sorted_indices]
    combined_s21 = combined_s21[sorted_indices]

    return combined_freq, combined_s11, combined_s21

def main():
    folder = r'c:\Users\VNA\Desktop\VNAF\June 27\Raw data\SB open'
    frequency, s11, s21 = combine_s2p_data(folder)

    # Interpolating to ensure uniform frequency spacing
    freq_min, freq_max = frequency.min(), frequency.max()
    num_points = len(frequency)
    uniform_freq = np.linspace(freq_min, freq_max, num_points)
    s11_interp = np.interp(uniform_freq, frequency, s11.real) + 1j * np.interp(uniform_freq, frequency, s11.imag)

    # Apply a Hanning window to reduce spectral leakage
    window = np.hanning(num_points)
    s11_windowed = s11_interp * window

    # Perform Inverse Fast Fourier Transform (IFFT)
    time_domain_data = np.fft.ifft(s11_windowed)

    # Calculate the time vector
    frequency_step = uniform_freq[1] - uniform_freq[0]
    time_vector = np.fft.fftfreq(num_points, d=frequency_step)

    # Plot the TDR result
    plt.figure(figsize=(12, 6))
    plt.plot(time_vector, np.abs(time_domain_data))
    plt.title("TDR Result from S11")
    plt.xlabel("Time (s)")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.show()

    return time_vector, np.abs(time_domain_data)

if __name__ == "__main__":
    time_vector, tdr_data = main()
    print(time_vector[:5], tdr_data[:5])
