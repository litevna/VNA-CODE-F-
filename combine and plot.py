import os
import pandas as pd
import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox
from scipy.signal import find_peaks
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from queue import Queue

# Global variables for plot and data
voltage_magnitude_s11 = None
voltage_magnitude_s21 = None
time_axis = None
folder_path = None
peak_label = None
root = None
queue = Queue()

# Near the beginning of your script or wherever the variables are defined
data_file_path = r"C:\Users\VNA\Desktop\VNAF\1 July\Excel Data\combined_data.csv"
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
    s11_peaks, _ = find_peaks(voltage_magnitude_s11, prominence=1)  # Adjust prominence if necessary
    s21_peaks, _ = find_peaks(voltage_magnitude_s21, prominence=1)  # Adjust prominence if necessary
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
    global peak_label
    peak_text = queue.get()
    if peak_label is not None:
        peak_label.setText(peak_text)

def update_plot():
    global voltage_magnitude_s11, voltage_magnitude_s21, time_axis, folder_path

    df, new_files = combine_files_in_folder(folder_path)
    if new_files:
        if df is not None:
            uniform_freq, voltage_magnitude_s11, voltage_magnitude_s21, time_axis = load_and_process_data(df.copy(), 1)
            plot_widget.clear()
            plot_widget.plot(time_axis, voltage_magnitude_s11, pen='r', name='s11 Magnitude')
            plot_widget.plot(time_axis, voltage_magnitude_s21, pen='g', name='s21 Magnitude')
            plot_widget.setLabel('bottom', 'Time (s)')
            plot_widget.setLabel('left', 'Voltage Magnitude')
            plot_widget.setTitle('Voltage Magnitude of s11 and s21 vs Time - Combined Data')
            plot_widget.showGrid(x=True, y=True)  # Enable grid lines
            auto_display_peaks(time_axis, voltage_magnitude_s11, voltage_magnitude_s21)
            display_peak_info()
        else:
            plot_widget.clear()
            plot_widget.setLabel('bottom', 'Time (s)')
            plot_widget.setLabel('left', 'Voltage Magnitude')
            plot_widget.setTitle('No Data Available')
            plot_widget.showGrid(x=True, y=True)  # Enable grid lines

def start_observer():
    global folder_path

    observer = Observer()
    observer.schedule(FileHandler(folder_path, lambda: queue.put('update')), folder_path)
    observer.start()

def main():
    app = QApplication([])
    global folder_path, root, plot_widget, peak_label

    win = pg.GraphicsLayoutWidget()
    win.setWindowTitle('VNA Data Visualization')
    plot_widget = win.addPlot()
    peak_label = win.addLabel()  # Add label for peak information
    folder_path = QFileDialog.getExistingDirectory(win, "Select folder containing S2P files")
    if folder_path:
        df, _ = combine_files_in_folder(folder_path)
        if df is not None:
            uniform_freq, voltage_magnitude_s11, voltage_magnitude_s21, time_axis = load_and_process_data(df.copy(), 1)
            plot_widget.plot(time_axis, voltage_magnitude_s11, pen='r', name='s11 Magnitude')
            plot_widget.plot(time_axis, voltage_magnitude_s21, pen='g', name='s21 Magnitude')
            plot_widget.setLabel('bottom', 'Time (s)')
            plot_widget.setLabel('left', 'Voltage Magnitude')
            plot_widget.setTitle('Voltage Magnitude of s11 and s21 vs Time - Combined Data')
            plot_widget.showGrid(x=True, y=True)  # Enable grid lines
            timer = QTimer()
            timer.timeout.connect(update_plot)
            timer.start(2000)  # Update plot every 2 seconds
            start_observer()
            win.show()
            app.exec_()
        else:
            QMessageBox.critical(win, "Error", f"No valid data found in folder: {folder_path}")
    else:
        QMessageBox.information(win, "Info", "Folder selection cancelled.")

if __name__ == "__main__":
    main()
