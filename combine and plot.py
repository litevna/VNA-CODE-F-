import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
from tkinter import Tk, messagebox
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

def combine_files_in_folder(folder_path):
    all_data = []
    input_folder_name = os.path.basename(folder_path)
    output_folder = os.path.join(folder_path, "Excel data")
    os.makedirs(output_folder, exist_ok=True)
    
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
        output_csv_path = os.path.join(output_folder, f"{input_folder_name}.csv")

        # Check if the file already exists
        if os.path.exists(output_csv_path):
            overwrite = messagebox.askyesno("File Exists", f"The file '{input_folder_name}.csv' already exists. Do you want to overwrite it?")
            if not overwrite:
                print("File not overwritten. Loading existing CSV file for plotting.")
                existing_df = pd.read_csv(output_csv_path)
                return existing_df
        
        combined_df.to_csv(output_csv_path, index=False)
        print(f"Combined data has been saved to {output_csv_path}")
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
    return uniform_freq, np.abs(s11_time), np.abs(s21_time), time_vector

def onselect(xmin, xmax):
    global voltage_magnitude_s11, voltage_magnitude_s21, time_axis, peak_details
    print(f"Selected range: {xmin} to {xmax}")
    indmin, indmax = np.searchsorted(time_axis, (xmin, xmax))
    indmin = max(0, indmin)
    indmax = min(len(time_axis) - 1, indmax)
    print(f"Indices: {indmin} to {indmax}")
    if indmax > indmin:
        selected_time_axis = time_axis[indmin:indmax]
        selected_s11 = voltage_magnitude_s11[indmin:indmax]
        selected_s21 = voltage_magnitude_s21[indmin:indmax]
        s11_peaks, _ = find_peaks(selected_s11, prominence=0.1)
        s21_peaks, _ = find_peaks(selected_s21, prominence=0.1)
        peak_text = ""
        if len(s11_peaks) > 0:
            peak_time_s11 = selected_time_axis[s11_peaks[0]]
            distance_s11 = peak_time_s11 * 9.891e7 / 2
            peak_text += f"s11 peak:\nTime: {peak_time_s11:.2e} s\nDistance: {distance_s11:.2e} meters\n"
        else:
            peak_text += "No significant s11 peaks found in the selected area\n"
        
        if len(s21_peaks) > 0:
            peak_time_s21 = selected_time_axis[s21_peaks[0]]
            distance_s21 = peak_time_s21 * 9.891e7 / 2
            peak_text += f"s21 peak:\nTime: {peak_time_s21:.2e} s\nDistance: {distance_s21:.2e} meters"
        else:
            peak_text += "No significant s21 peaks found in the selected area"
        
        peak_details.set_text(peak_text)
        print(peak_text)
    else:
        print("No significant peaks found in the selected area")

def main():
    global voltage_magnitude_s11, voltage_magnitude_s21, time_axis, peak_details

    root = Tk()
    root.withdraw()
    folder_path = askdirectory(title="Select folder containing S2P files")
    if folder_path:
        df = combine_files_in_folder(folder_path)
        if df is not None:
            uniform_freq, voltage_magnitude_s11, voltage_magnitude_s21, time_axis = load_and_process_data(df.copy(), 1)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
            ax1.plot(time_axis, voltage_magnitude_s11, label='s11 Magnitude')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Voltage Magnitude')
            ax1.set_title('Voltage Magnitude of s11 vs Time - Combined Data')
            ax1.legend()
            ax1.grid(which='both')
            ax1.minorticks_on()
            ax2.plot(time_axis, voltage_magnitude_s21, label='s21 Magnitude')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Voltage Magnitude')
            ax2.set_title('Voltage Magnitude of s21 vs Time - Combined Data')
            ax2.legend()
            ax2.grid(which='both')
            ax2.minorticks_on()
            max_y_s11 = np.max(voltage_magnitude_s11) * 1.1
            ax1.set_ylim(0, max_y_s11)
            max_y_s21 = np.max(voltage_magnitude_s21) * 1.1
            ax2.set_ylim(0, max_y_s21)
            peak_details = ax1.text(0.5, 0.95, "", transform=ax1.transAxes, ha='center', va='top', bbox=dict(facecolor='white', alpha=0.5))
            span_s11 = SpanSelector(ax1, onselect, 'horizontal', useblit=True, span_stays=True)
            span_s21 = SpanSelector(ax2, onselect, 'horizontal', useblit=True, span_stays=True)
            plt.tight_layout()
            plt.show()
    else:
        print("No folder selected.")

if __name__ == "__main__":
    main()
