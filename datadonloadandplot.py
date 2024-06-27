import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Function to read Touchstone (S2P) file
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

# Function to select S2P file using a dialog box
def select_file():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    filepath = filedialog.askopenfilename(title="Select S2P File")
    return filepath

# Function to find peaks within a user-specified time range
def find_peaks_in_area(time_vector, magnitude_data, xlim):
    # Identify indices corresponding to the selected time range
    start_index = np.searchsorted(time_vector, xlim[0])
    end_index = np.searchsorted(time_vector, xlim[1])

    # Extract data within the selected time range
    selected_data = magnitude_data[start_index:end_index]
    selected_time = time_vector[start_index:end_index]

    # Find peak locations using SciPy's find_peaks function
    peaks, _ = find_peaks(selected_data)

    # Return peak locations and corresponding magnitudes within the selected area
    return selected_time[peaks], selected_data[peaks]

# Function for handling user clicks on the plot for interactive selection
def on_click(event):
    if event.xdata is not None:  # Ensure a valid click within the plot area
        global xlim  # Access the global variable storing the selected area

        # Update the selected area based on the click position
        xlim[0] = min(xlim[0], event.xdata)  # Update minimum time
        xlim[1] = max(xlim[1], event.xdata)  # Update maximum time

        # Update the plot to reflect the changed selected area
        ax.set_xlim(xlim)

        # Find and highlight peaks within the new selected area
        selected_peaks_time, selected_peaks_data = find_peaks_in_area(time_vector, magnitude_data, xlim)
        ax.scatter(selected_peaks_time, selected_peaks_data, c='red', label='Selected Peaks')

        # Print peak information if any are found within the selected area
        if len(selected_peaks_time) > 0:
            print(f"Selected Peak Locations (Time): {selected_peaks_time}")
            print(f"Selected Peak Magnitudes: {selected_peaks_data}")

        # Update plot legend
        ax.legend()

        # Trigger a redraw of the plot with the updated elements
        fig.canvas.draw()

# Main program execution
if __name__ == "__main__":
    import os  # Added import

    # Open a file selection dialog to choose an S2P file
    filepath = select_file()

    if filepath:
        # Process the selected S2P file
        frequency, s11, s21 = read_s2p_file(filepath)

        # TDR processing section
        freq_min, freq_max = frequency.min(), frequency.max()
        num_points = len(frequency)
        uniform_freq = np.linspace(freq_min, freq_max, num_points)
        s11_interp = np.interp(uniform_freq, frequency, s11.real) + 1j * np.interp(uniform_freq, frequency, s11.imag)

        window = np.hanning(num_points)
        s11_windowed = s11_interp * window

        time_domain_data = np.fft.ifft(s11_windowed)

        frequency_step = uniform_freq[1] - uniform_freq[0]
        time_vector = np.fft.fftfreq(num_points, d=frequency_step)

        # Global variable to store the selected time range (initially set to entire plot)
        global xlim
        xlim = [time_vector[0], time_vector[-1]]

        # Create the Matplotlib figure and plot the TDR magnitude
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(time_vector, np.abs(time_domain_data))
        ax.set_title(f"TDR Result from {filepath}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Magnitude")
        ax.set_xlim(xlim)  # Set initial x-axis limits

        # Connect the function `on_click` to mouse button press events
        fig.canvas.mpl_connect('button_press_event', on_click)

        # Display the plot and wait for user interaction
        plt.show()
 
