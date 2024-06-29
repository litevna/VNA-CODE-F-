import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from tkinter import Tk, messagebox, Toplevel, Label, LEFT
from tkinter.filedialog import askdirectory
from scipy.signal import find_peaks
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Global variables for plot and data
voltage_magnitude_s11 = None
voltage_magnitude_s21 = None
time_axis = None
ax1 = None
ax2 = None
fig = None
folder_path = None


class FileHandler(FileSystemEventHandler):
    def __init__(self, folder_path, callback):
        super().__init__()
        self.folder_path = folder_path
        self.callback = callback

    def on_created(self, event):
        if event.src_path.endswith('.s2p'):
            self.callback()


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


def auto_display_peaks(time_axis, voltage_magnitude_s11, voltage_magnitude_s21):
    s11_peaks, _ = find_peaks(voltage_magnitude_s11, prominence=0.1)
    s21_peaks, _ = find_peaks(voltage_magnitude_s21, prominence=0.1)
    peak_text = ""
    if len(s11_peaks) > 0:
        peak_time_s11 = time_axis[s11_peaks[0]]
        distance_s11 = peak_time_s11 * 9.891e7 / 2
        peak_text += f"s11 peak:\nTime: {peak_time_s11:.2e} s\nDistance: {distance_s11:.2e} meters\n\n"
    if len(s21_peaks) > 0:
        peak_time_s21 = time_axis[s21_peaks[0]]
        distance_s21 = peak_time_s21 * 9.891e7 / 2
        peak_text += f"s21 peak:\nTime: {peak_time_s21:.2e} s\nDistance: {distance_s21:.2e} meters\n\n"
    display_peak_info(peak_text)


def display_peak_info(peak_text):
    global fig
    if fig is not None:
        if not hasattr(display_peak_info, 'peak_window') or display_peak_info.peak_window is None:
            # Execute Tkinter operations within the main loop
            fig.canvas.manager.window.after(100, lambda: display_peak_info(peak_text))
        else:
            # Update existing peak window content
            display_peak_info.peak_window.attributes('-topmost', True)
            peak_label.config(text=peak_text)
    else:
        print("Plot window is not initialized yet.")


def update_plot():
    global voltage_magnitude_s11, voltage_magnitude_s21, time_axis, ax1, ax2, fig, folder_path

    df = combine_files_in_folder(folder_path)
    if df is not None:
        uniform_freq, voltage_magnitude_s11, voltage_magnitude_s21, time_axis = load_and_process_data(df.copy(), 1)
        ax1.cla()
        ax2.cla()
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
        fig.canvas.draw_idle()
        auto_display_peaks(time_axis, voltage_magnitude_s11, voltage_magnitude_s21)


def main():
    global ax1, ax2, fig, folder_path

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
            observer = Observer()
            observer.schedule(FileHandler(folder_path, update_plot), folder_path)
            observer.start()
            plt.tight_layout()
            plt.show()
            root.mainloop()
        else:
            messagebox.showerror("Error", f"No valid data found in folder: {folder_path}")
    else:
        messagebox.showinfo("Info", "Folder selection cancelled.")


if __name__ == "__main__":
    main()
