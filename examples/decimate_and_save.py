#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example use of simpleDASreader8.

The objective of this file is to illustrate basic processing of data 
using scipy.signal and saving to file
"""

import simpleDASreader
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sps
import datetime,os


#%% Find input files from experiment input folder and time interval
input_folder = '/raid1/fsi/exps/DASdisplay_demo_files/FrequencySweep/'

start = datetime.datetime(2021, 5, 31, 5, 44, 0)
duration = datetime.timedelta(seconds=18)


# Request a subset of channels. The function below will inspect the file and 
# determine which channels exist in the file, returned as the chIndex variable
channels = np.r_[np.arange(7500, 8000, 5),np.arange(8500, 9000, 5)]

file_names, chIndex, samples = simpleDASreader.find_DAS_files(input_folder, start, duration,
                                                              channels=channels,load_file_from_start=False)

#%% Load the data files for the channels requested and found

signal = simpleDASreader.load_DAS_files(file_names, chIndex, samples)

#%% Show the first five columns to check the names of the series and data type

print(signal.head(5))


#%% decimate signal

decimation_factor = 100
Nt_in = signal.shape[0]
# Alternative decimation methods. NB! these methods returns ndarray not DataFrame, even for DataFrame input
#sig_decimated = sps.resample(signal,Nt_in//decimation_factor) # resample in frequency domain
#sig_decimated = sps.decimate(signal, decimation_factor,ftype='fir',axis=0) # resample with filter
sig_decimated = sps.resample_poly(signal,up=1,down=decimation_factor,axis=0,padtype='edge') # resample with polyphase filter implementation

Nt_out = sig_decimated.shape[0]
dt_out = signal.meta['dt']*Nt_in/Nt_out
tstart = signal.meta['time'] + datetime.timedelta(seconds=0) # may add a timeoffset to adjust for processing delay
t = simpleDASreader.create_time_axis(tstart,sig_decimated.shape[0],dt_out)
meta_out= signal.meta.copy()
meta_out.update(dt= dt_out,time=tstart)
signal_decimated = simpleDASreader.DASDataFrame(sig_decimated,index=t,columns=signal.columns,
                                               meta = meta_out)

#%% save output

filename_out = simpleDASreader.save_to_DAS_file(signal_decimated)

plt.figure(1,clear=True)
plt.plot(signal[7500],label='Undecimated')
plt.plot(signal_decimated[7500],label='Undecimated')
import matplotlib.dates as mdates
plt.gca().xaxis.set_minor_formatter(mdates.DateFormatter("%M:%s"))
plt.legend()

#%% reload data

file_names, chIndex, samples = simpleDASreader.find_DAS_files(input_folder, start, duration,datatype='processed')
signal_reloaded = simpleDASreader.load_DAS_files(file_names, chIndex, samples)
