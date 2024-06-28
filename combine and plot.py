import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askdirectory
from scipy.signal import find_peaks

def process_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    data_start_index = 0
    for i, line in enumerate(lines):
        if line.strip() and (line[0].isdigit() or line[0] == '!'):
            data_start_index = i
            break
    data_lines = lines[data_start_index:]
    data = []
    for line in data_lines:
        if line.startswith('!'):
            line = line[1:]
        columns = line.split()
        if len(columns) >= 5:
            try:
                frequency = float(columns[0])
                s11_real = float(columns[1])
                s11_imag = float(columns[2])
                s21_real = float(columns[3])
                s21_imag = float(columns[4])
                data.append([frequency, s11_real, s11_imag, s21_real, s21_imag])
            except ValueError:
                print(f"Skipping line with invalid data: {line.strip()}")
    if data:
        df = pd.DataFrame(data, columns=['frequency', 's11 real', 's11 imag', 's21 real', 's21 imag'])
        return df
    else:
        return None

def combine_files_in_folder(folder_path, output_path, output_filename):
    all_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.s2p'):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing file: {file_path}")
            df = process_file(file_path)
            if df is not None:
                all_data.append(df)
            else:
                print(f"No data extracted from file: {file_path}")
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        output_file_path = os.path.join(output_path, output_filename + '.csv')
        combined_df.to_csv(output_file_path, index=False)
        print(f"Combined data has been saved to {output_file_path}")
        return combined_df
    else:
        return None

def load_and_process_data(df, num_sets):
    df = df.dropna()
    total_points = 1024 * num_sets
    df = df.head(total_points)
    frequencies = df['frequency'].values
    s11_real = df['s11 real'].values
    s11_imag = df['s11 imag'].values
    s21_real = df['s21 real'].values
    s21_imag = df['s21 imag'].values
    s11_complex = s11_real + 1j * s11_imag
    s21_complex = s21_real + 1j * s21_imag
    freq_min, freq_max = frequencies.min(), frequencies.max()
    num_points = len(frequencies)
    uniform_freq = np.linspace(freq_min, freq_max, num_points)
    s11_interp = np.interp(uniform_freq, frequencies, s11_complex.real) + 1j * np.interp(uniform_freq, frequencies, s11_complex.imag)
    s21_interp = np.interp(uniform_freq, frequencies, s21_complex.real) + 1j * np.interp(uniform_freq, frequencies, s21_complex.imag)
    window = np.hanning(num_points)
    s11_windowed = s11_interp * window
    s21_windowed = s21_interp * window
    s11_time = np.fft.ifft(s11_windowed)
    s21_time = np.fft.ifft(s21_windowed)
    frequency_step = uniform_freq[1] - uniform_freq[0]
    time_vector = np.fft.fftfreq(num_points, d=frequency_step)
    s11_peaks, _ = find_peaks(np.abs(s11_time), prominence=0.1)
    s21_peaks, _ = find_peaks(np.abs(s21_time), prominence=0.1)
    if len(s11_peaks) > 0:
        peak_time_s11 = time_vector[s11_peaks[0]]
        distance_s11 = peak_time_s11 * 9.891e7 / 2
    else:
        peak_time_s11 = None
        distance_s11 = None
    if len(s21_peaks) > 0:
        peak_time_s21 = time_vector[s21_peaks[0]]
        distance_s21 = peak_time_s21 * 9.891e7 / 2
    else:
        peak_time_s21 = None
        distance_s21 = None
    return uniform_freq, np.abs(s11_time), np.abs(s21_time), time_vector, peak_time_s11,distance_s11, peak_time_s21, distance_s21

def main():
    root = Tk()
    root.withdraw()
    folder_path = askdirectory(title="Select folder containing S2P files")
    if folder_path:
        output_path = folder_path
        output_filename = "combined_data"
        df = combine_files_in_folder(folder_path, output_path, output_filename)
        if df is not None:
            uniform_freq, voltage_magnitude_s11, voltage_magnitude_s21, time_axis, peak_time_s11, distance_s11, peak_time_s21, distance_s21 = load_and_process_data(df.copy(), 1)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
            ax1.plot(time_axis, voltage_magnitude_s11, label='s11 Magnitude')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Voltage Magnitude')
            ax1.set_title('Voltage Magnitude of s11 vs Time')
            ax1.legend()
            ax1.grid(which='both')
            ax1.minorticks_on()
            ax2.plot(time_axis, voltage_magnitude_s21, label='s21 Magnitude')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Voltage Magnitude')
            ax2.set_title('Voltage Magnitude of s21 vs Time')
            ax2.legend()
            ax2.grid(which='both')
            ax2.minorticks_on()
            max_y_s11 = np.max(voltage_magnitude_s11) * 1.1
            ax1.set_ylim(0, max_y_s11)
            max_y_s21 = np.max(voltage_magnitude_s21) * 1.1
            ax2.set_ylim(0, max_y_s21)
            if peak_time_s11 is not None:
                print(f"s11 peak: {peak_time_s11:.2e} s, Distance: {distance_s11:.2e} meters")
            else:
                print("No significant s11 peaks found")
            if peak_time_s21 is not None:
                print(f"s21 peak: {peak_time_s21:.2e} s, Distance: {distance_s21:.2e} meters")
            else:
                print("No significant s21 peaks found")
            plt.tight_layout()
            plt.show()
    else:
        print("No folder selected.")

if __name__ == "__main__":
    main()