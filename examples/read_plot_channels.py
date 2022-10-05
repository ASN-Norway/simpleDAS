#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example use of simpleDASreader.

The objective of this file is to illustrate basic reading and investigation
of OptoDAS measurements.
"""

import simpleDASreader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.dates as mdates
import scipy.signal as sps
import datetime,os


#%% Find input files from experiment input folder and time interval

input_folder = '/raid1/fsi/exps/DASdisplay_demo_files/FrequencySweep/'

start = datetime.datetime(2021, 5, 31, 5, 44, 0)
duration = datetime.timedelta(seconds=18)

# Request a subset of channels. The function below will inspect the file and 
# determine which channels exist in the file, returned as the chIndex variable
channels = np.arange(7500, 10500, 5)

file_names, chIndex, samples = simpleDASreader.find_DAS_files(input_folder, start, duration,
                                                              channels=channels,load_file_from_start=False)

#%% Load the data files for the channels requested and found

signal = simpleDASreader.load_DAS_files(file_names, chIndex, samples)
#%% Show the first five columns to check the names of the series and data type

print(signal.head(5))

#%% Plot the data magntitude for time and channel

dt = signal.meta['dt']
Nt = len(signal)
plt.figure(1,clear=True)
plt.imshow(np.abs(signal), norm=colors.LogNorm(vmin=1e-9),
           extent=[signal.columns[0], signal.columns[-1], len(signal), 0])
plt.colorbar()

#%% Compare the time series data for few channels
plt.figure(2,clear=True)
for ch in range(7500, 9000, 500):
    plt.plot(signal[ch], label=str(ch))
plt.legend()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%M:%S"))
#Alternative 

plt.figure(2,clear=True); ax = plt.gca()
chs = np.arange(7500, 9000, 500)
plt.plot(signal.loc[:start+datetime.timedelta(seconds=3.),chs])
plt.legend(['Ch %d' %ch for ch in chs])
plt.xlabel('Time')
plt.ylabel('Signal [%s]' % signal.meta['unit'])
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%M:%S"))

#%% Compare the PSD for the same channels

plt.figure(3,clear=True)
for ch in range(7500, 9000, 500):
    f,Pxx = sps.welch(signal[ch],fs=1/signal.meta['dt'],axis=0)
    plt.loglog(f,np.sqrt(Pxx), label=f'Ch {ch}')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude spectral density [%s/√Hz]'% signal.meta['unit'])
plt.legend()
plt.tight_layout()
#%% Export the first 3 seconds of channel data to a csv file

signal[: start+datetime.timedelta(seconds=3.)].to_csv('das_data.csv')



