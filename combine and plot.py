import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, messagebox, Toplevel, Label, LEFT
from tkinter.filedialog import askdirectory
from scipy.signal import find_peaks
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from matplotlib.animation import FuncAnimation
from queue import Queue

# Global variables for plot and data
voltage_magnitude_s11 = None
voltage_magnitude_s21 = None
time_axis = None
ax = None
fig = None
folder_path = None
peak_label = None
root = None
queue = Queue()
data_file_path = "combined_data.csv"
processed_files = set()

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
    global processed_files
    all_data = []
    new_files = False
    for filename in os.listdir(folder_path):
        if filename.endswith('.s2p') and filename not in processed_files:
            file_path = os.path.join(folder_path, filename)
            print(f"Processing file: {file_path}")
            df = process_file(file_path)
            if df is not None:
                all_data.append(df)
                processed_files.add(filename)
                new_files = True
            else:
                print(f"No data extracted from file: {file_path}")
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df.to_csv(data_file_path, index=False)  # Save combined data to CSV
        return combined_df, new_files
    else:
        return None, new_files

def load_and_process_data(df, num_sets):
    df = df.dropna()
    total_points = min(1024 * num_sets, len(df))
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
    s11_peaks, _ = find_peaks(voltage_magnitude_s11, prominence=0.05)
    s21_peaks, _ = find_peaks(voltage_magnitude_s21, prominence=0.05)
    peak_text = ""
    if len(s11_peaks) > 0:
        peak_time_s11 = time_axis[s11_peaks[0]]
        distance_s11 = peak_time_s11 * 9.891e7 / 2
        peak_text += f"s11 peak:\nTime: {peak_time_s11:.2e} s\nDistance: {distance_s11:.2e} meters\n\n"
    if len(s21_peaks) > 0:
        peak_time_s21 = time_axis[s21_peaks[0]]
        distance_s21 = peak_time_s21 * 9.891e7 / 2
        peak_text += f"s21 peak:\nTime: {peak_time_s21:.2e} s\nDistance: {distance_s21:.2e} meters\n\n"
    queue.put(peak_text)

def display_peak_info():
    global fig, peak_label
    peak_text = queue.get()
    if fig is not None:
        if not hasattr(display_peak_info, 'peak_window') or display_peak_info.peak_window is None:
            peak_window = Toplevel()
            peak_window.title("Peak Information")
            peak_label = Label(peak_window, text=peak_text, justify=LEFT)
            peak_label.pack()
            display_peak_info.peak_window = peak_window
        else:
            display_peak_info.peak_window.attributes('-topmost', True)
            peak_label.config(text=peak_text)
    else:
        print("Plot window is not initialized yet.")

def update_plot(frame):
    global voltage_magnitude_s11, voltage_magnitude_s21, time_axis, ax, fig, folder_path

    df, new_files = combine_files_in_folder(folder_path)
    if new_files:
        if df is not None:
            uniform_freq, voltage_magnitude_s11, voltage_magnitude_s21, time_axis = load_and_process_data(df.copy(), 1)
            ax.clear()
            ax.plot(time_axis, voltage_magnitude_s11, label='s11 Magnitude')
            ax.plot(time_axis, voltage_magnitude_s21, label='s21 Magnitude')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Voltage Magnitude')
            ax.set_title('Voltage Magnitude of s11 and s21 vs Time - Combined Data')
            ax.legend()
            ax.grid(which='both')
            ax.minorticks_on()
            max_y = np.max([np.max(voltage_magnitude_s11), np.max(voltage_magnitude_s21)]) * 1.1
            ax.set_ylim(0, max_y)
            ax.set_xlim(0, time_axis.max() * 0.1)  # Adjust to focus on the relevant area
            auto_display_peaks(time_axis, voltage_magnitude_s11, voltage_magnitude_s21)
        else:
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Voltage Magnitude')
            ax.set_title('No Data Available')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.text(0.5, 0.5, 'No valid data found', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

def start_observer():
    global folder_path

    observer = Observer()
    observer.schedule(FileHandler(folder_path, lambda: queue.put('update')), folder_path)
    observer.start()

def main():
    global ax, fig, folder_path, root

    root = Tk()
    root.withdraw()
    folder_path = askdirectory(title="Select folder containing S2P files")
    if folder_path:
        df, _ = combine_files_in_folder(folder_path)
        if df is not None:
            uniform_freq, voltage_magnitude_s11, voltage_magnitude_s21, time_axis = load_and_process_data(df.copy(), 1)
            fig, ax = plt.subplots(figsize=(10, 6))
            ani = FuncAnimation(fig, update_plot, interval=2000)  # Update plot every 2 seconds
            plt.show()
            root.mainloop()
        else:
            messagebox.showerror("Error", f"No valid data found in folder: {folder_path}")
    else:
        messagebox.showinfo("Info", "Folder selection cancelled.")

if __name__ == "__main__":
    main()
