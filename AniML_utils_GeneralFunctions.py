"""
ARBEL
a 'storage' of general functions used in the script.
Not all are used.
"""
from scipy.ndimage import convolve1d
import scipy.signal as signal
import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from datetime import datetime

def open_file_as_dataframe(file_path):
    # Get the file extension
    file_extension = file_path.split('.')[-1]
    # Open the file based on the extension
    if file_extension == 'csv':
        df = pd.read_csv(file_path,  dtype='a')
        new_headers = df.iloc[0, 1:] + '_' + df.iloc[1, 1:]
        new_headers = new_headers.replace('likelihood', 'prob', regex=True)
        df = df.iloc[2:, 1:]
        df.columns = new_headers
        df = df.astype(float)
    elif file_extension == 'h5':
        df = pd.read_hdf(file_path)
        current_headers = df.columns # Get the current column names
        new_headers = [] # Create a new list to store the updated column names
        for column_name in current_headers:
            new_header = column_name[1:] # Drop the first level of the MultiIndex column name
            new_header = '_'.join(new_header) # Update the column name with the desired pattern
            new_headers.append(new_header)
        df.columns = new_headers # Update the column names using the updated_column_names list
        df.columns = df.columns.str.replace('likelihood', 'prob')
    else:
        raise ValueError("Unsupported file format. Only CSV and H5 files are supported.")
    return df

def moving_window_filter(df, window_size, sigma):
    filtered_df = df.copy()  # Create a copy of the original DataFrame
    num_columns = len(df.columns)
    x = np.arange(-window_size // 2 + 1, window_size // 2 + 1)
    kernel = np.exp(-x ** 2 / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)
    # Normalize the kernel
    kernel /= np.sum(kernel)

    # Iterate over the columns
    for i in range(num_columns):
        column_values = df.iloc[:, i]  # Get the values of the current column
        # Apply the filter using convolution
        filtered_df.iloc[:, i] = convolve1d(column_values, kernel, mode='constant')
    return filtered_df

def binary_gaussian_moving_window_filter(signal, sigma, window_size):
    # Define the Gaussian kernel
    x = np.arange(-window_size // 2 + 1, window_size // 2 + 1)
    kernel = np.exp(-x ** 2 / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)

    # Normalize the kernel
    kernel /= np.sum(kernel)

    # Apply the filter using convolution
    filtered_signal = convolve1d(signal, kernel, mode='constant')

    # Threshold the filtered signal to binary (0/1)
    threshold = 0.5
    filtered_signal = (filtered_signal > threshold).astype(int)
    return filtered_signal

def switch_columns(dataframe, col1, col2):
    # Make sure the specified columns are in the DataFrame
    if col1 in dataframe.columns and col2 in dataframe.columns:
        dataframe[col1], dataframe[col2] = dataframe[col2].values, dataframe[col1].values
    else:
        print(f"One or both of the specified columns ({col1}, {col2}) not found in DataFrame.")

def peak_freq(signal_intensity, window_size = 25, step_size = 1):
    peak_frequencies = []
    for i in range(0, len(signal_intensity) - window_size + 1, step_size):
        frame = signal_intensity[i:i + window_size]
        # Perform FFT on the frame
        frequencies, power_spectrum = signal.welch(frame, fs=25, nperseg=window_size)
        # Find the peak frequency in the power spectrum
        peak_frequency = frequencies[np.argmax(power_spectrum)]
        peak_frequencies.append(peak_frequency)
    peak_frequencies= np.concatenate((np.array(peak_frequencies,dtype=int), np.full(window_size-1, np.nan)))

    # plt.figure(figsize=(10, 5))
    # plt.plot(peak_frequencies)
    # plt.xlabel('Frame')
    # plt.ylabel('Peak Frequency (Hz)')
    # plt.title('Peak Frequency Frame by Frame')
    # plt.grid(True)
    # plt.show()
    return peak_frequencies

def flip_video(input_folder, output_folder='', new_file_ending='_flipped'):
    output_path=input_folder+'/'+output_folder
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Get the list of AVI files in the input folder
    avi_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.avi')])

    for avi_file in avi_files:
        # Read the video
        print(f'Flipping {avi_file}')
        video_path = os.path.join(input_folder, avi_file)
        cap = cv2.VideoCapture(video_path)

        # Get video properties
        width = int(cap.get(3))
        height = int(cap.get(4))
        fps = cap.get(5)

        # Create VideoWriter for flipped video
        output_file = os.path.join(output_path, os.path.splitext(avi_file)[0] + new_file_ending +'.avi')
        out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
        print(f"Output: {os.path.join(output_path, os.path.splitext(avi_file)[0] + new_file_ending +'.avi')}")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Flip the frame left-to-right
            flipped_frame = cv2.flip(frame, 1)

            # Write the flipped frame to the output video
            out.write(flipped_frame)

        # Release video capture and writer objects
        cap.release()
        out.release()

    print("Videos flipped and saved to:", output_path)

def min_consecutive_ones(series, min_ones):
    transformed_series = np.zeros(len(series))
    consecutive_ones_count = 0
    series=np.ravel(series)
    for i in range(0, len(series)):
        if series[i] == 1:
            transformed_series[i] = 1
            consecutive_ones_count += 1
        if series[i] == 0:
            transformed_series[i] = 0
            if consecutive_ones_count < min_ones and i>0:
                transformed_series[i-1-consecutive_ones_count:i] = 0
            consecutive_ones_count = 0
    return transformed_series.astype(int)

def closeall():
    plt.close("all")

def clc():
    print(50*"\n")

def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print ("Elapsed time is " + str(round(time.time() - startTime_for_tictoc,2)) + " seconds.")
    else:
        print ("Toc: start time not set")

def Done(*args):
    print('done!')
    # import winsound
    # for i in range(0,3):
    #     frequency = 3700  # Set Frequency To 2500 Hertz
    #     duration = 50  # Set Duration To 1000 ms == 1 second
    #     winsound.Beep(frequency, duration)
    #     frequency = 1000  # Set Frequency To 2500 Hertz
    #     duration = 50  # Set Duration To 1000 ms == 1 second
    #     winsound.Beep(frequency, duration)