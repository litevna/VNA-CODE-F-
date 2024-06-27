import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, Button, Label, StringVar
from tkinter.filedialog import askdirectory
from scipy.signal import find_peaks
from matplotlib.widgets import RectangleSelector
import threading

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

    s11_time = np.fft.ifft(s11_complex)
    s21_time = np.fft.ifft(s21_complex)

    voltage_magnitude_s11 = np.abs(s11_time)
    voltage_magnitude_s21 = np.abs(s21_time)

    num_points = len(frequencies)
    frequency_step = 2.052786e9  # 2.052786 GHz
    time_step = 1 / frequency_step  # Time step corresponding to the frequency step
    time_axis = np.fft.fftfreq(num_points, d=time_step)

    return frequencies, voltage_magnitude_s11, voltage_magnitude_s21, time_axis

def copy_to_clipboard(event, text):
    r = Tk()
    r.withdraw()
    r.clipboard_clear()
    r.clipboard_append(text)
    r.update()
    r.destroy()

def onselect(eclick, erelease):
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, y1  # Ensuring selection height is zero, as we are interested in x-axis selection only

    if x1 > x2:
        x1, x2 = x2, x1

    mask = (time_axis >= x1) & (time_axis <= x2)
    selected_s11 = voltage_magnitude_s11[mask]
    selected_s21 = voltage_magnitude_s21[mask]
    selected_time = time_axis[mask]

    if len(selected_s11) > 0 and len(selected_s21) > 0:
        peak_index_s11 = np.argmax(selected_s11)
        peak_index_s21 = np.argmax(selected_s21)

        peak_time_s11 = selected_time[peak_index_s11]
        peak_time_s21 = selected_time[peak_index_s21]

        highest_peak_s11 = (peak_time_s11, selected_s11[peak_index_s11])
        highest_peak_s21 = (peak_time_s21, selected_s21[peak_index_s21])

        time_difference = abs(highest_peak_s11[0] - highest_peak_s21[0])
        speed_of_signal = 9.891e7  # Speed of signal in meters per second
        distance = time_difference * speed_of_signal / 2

        details = (
            f"s11 peak: {highest_peak_s11[0]:.2e} s, s21 peak: {highest_peak_s21[0]:.2e} s, "
            f"Time difference: {time_difference:.2e} s, Distance between peaks: {distance:.2e} meters"
        )
        print(details)

        ax.plot(highest_peak_s11[0], highest_peak_s11[1], 'ro', label='Peak of s11')
        ax.plot(highest_peak_s21[0], highest_peak_s21[1], 'bo', label='Peak of s21')
        ax.annotate(details, xy=(0.5, -0.1), xycoords='axes fraction', ha='center', fontsize=8)
        fig.canvas.draw()

        details_var.set(details)
        copy_button.config(command=lambda: copy_to_clipboard(None, details))

def confirm_selection():
    print(details_var.get())

def auto_close_window():
    root.after(300000, root.destroy)  # Close the Tkinter window after 300 seconds (300,000 milliseconds)

root = Tk()
folder_path = askdirectory(title="Select Folder Containing .s2p Files")

if not folder_path:
    print("No folder selected. Exiting.")
else:
    output_folder_path = r'c:\Users\VNA\Desktop\VNAF\26 June\Excel data'
    output_filename = os.path.basename(folder_path)

    combined_df = combine_files_in_folder(folder_path, output_folder_path, output_filename)

    if combined_df is not None:
        num_sets = combined_df.shape[0] // 1024  # Calculate number of sets in the combined data
        print(f"Number of data sets: {num_sets}")
        
        num_sets_to_plot = int(input("Enter the number of sets of data to plot: "))
        data = load_and_process_data(combined_df, num_sets_to_plot)

        if data is not None:
            frequencies, voltage_magnitude_s11, voltage_magnitude_s21, time_axis = data

            fig, ax = plt.subplots(figsize=(12, 8))
            ax.plot(time_axis, voltage_magnitude_s11, label='s11 Magnitude')
            ax.plot(time_axis, voltage_magnitude_s21, label='s21 Magnitude')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Voltage Magnitude')
            ax.set_title(f'Voltage Magnitude of s11 and s21 vs Time {output_filename} - {num_sets_to_plot}')
            ax.legend()
            ax.grid(which='both')
            ax.minorticks_on()

            max_y = max(np.max(voltage_magnitude_s11), np.max(voltage_magnitude_s21)) * 1.1
            ax.set_ylim(0, max_y)

            rect_selector = RectangleSelector(ax, onselect, interactive=True, button=[1])

            details_var = StringVar()
            details_label = Label(root, textvariable=details_var)
            details_label.pack()

            copy_button = Button(root, text="Copy Details")
            copy_button.pack()

            confirm_button = Button(root, text="Confirm Selection", command=confirm_selection)
            confirm_button.pack()

            auto_close_window()

            plt.show()
