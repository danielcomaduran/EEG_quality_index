## EQI Quick Comparison
# - This script runs the EEG Quality Index to comparte two datasets
#   similar to the processing done in the 04_BCI_Move_Validation.ipynb

import matplotlib


# %matplotlib qt

#%% Import libraries
import os
import mne
import sys
import pyxdf
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.fft as fft
import scipy.stats as stats
import matplotlib.pyplot as plt
sys.path.append("..") # Adds higher directory to python modules path.
import eeg_quality_index
import import_data

#%% Import data
# - Notes
# -- Change data folder to match your local data folder
# -- Change the name of the clean and test data depending on which is the XDF or EDF file
# -- If both files (i.e., clean and test) are the same file tipe (i.e., EDF or XDF), use the appropriate method to import

# - Get data directory
current_directory = os.getcwd()
os.chdir("..")
data_folder = os.getcwd() + '\\Data\\'  # Change this if needed
os.chdir(current_directory)

# - Create MNE object from XDF data
# stream = 1  # Select proper stream from XDF file
# trial_name = '\\220621 EKL Gtec\\baseline1.xdf'  # Change this if needed
# print(f'{data_folder+trial_name}')
# streams, header = pyxdf.load_xdf(data_folder+trial_name)
# sfreq = float(streams[stream]['info']['nominal_srate'][0])                  # Sampling frequency [Hz]
# n_chans = len(streams[stream]['info']['desc'][0]['channels'][0]['channel']) # Number of channels [n]
# chans = [streams[stream]['info']['desc'][0]['channels'][0]['channel'][i]['label'][0] for i in range(n_chans)]   # Channel names
# info = mne.create_info(chans, sfreq, ch_types='eeg')
# data = streams[stream]['time_series'].T # EEG data [uV]
# clean = mne.io.RawArray(data*1e-9, info, verbose=False) # Change this variable name if needed

# # - Import EDF data to MNE object
# trial_name = '\\220607 Baseline Emotiv Flex\\PT_06.07.22_RS_EPOCFLEX_99556_2022.06.07T17.27.25.06.00.edf'   # Change this if needed
# test = mne.io.read_raw_edf(data_folder+trial_name, verbose=False, preload=True)

# - Import open BCI .TXT data
txt_file = "C:\\Users\\danie\\Documents\\Projects\\EEG_quality_index\\Data\\OpenBCI\\OpenBCI-RAW-2022-07-15_14-23-04.txt"
openBCI_srate = 125 # [Hz]
openBCI_eeg = import_data.import_openBCI(txt_file)

#%% Filter data
# - Select cutoff frequencies for bandpass filter and filter the data
fc_high = 40    # cutoff frequency high [Hz]
fc_low = 0.1    # cutoff frequency low [Hz]

# - Enable as needed
# -- Clean data MUST be filtered to avoid NaN results
clean.filter(l_freq=fc_low, h_freq=fc_high)
test.filter(l_freq=fc_low, h_freq=fc_high)

#%% Plot RAW data
# - Enable each line if needed
# - Use the help button in the plot figure for navigation
# clean.plot()
# test.plot()

#%% Pick times
# - Select times to trim the data
clean_trim_start = 45     # Time to begin trimming [sec]
clean_trim_end = 75       # Time to stop trimming [sec]

test_trim_start = 45     # Time to begin trimming [sec]
test_trim_end = 75       # Time to stop trimming [sec]

# - Enable as needed
clean.crop(tmin=clean_trim_start, tmax=clean_trim_end)
test.crop(tmin=test_trim_start, tmax=test_trim_end)

#%% Channel names
clean_chans = clean.info['ch_names']
print('Clean dataset channel names:')
print(f'{clean_chans}')

test_chans = test.info['ch_names']
print('\nTest  dataset channel names:')
print(f'{test_chans}')

#%% Pick channels
# - Edit these variables to pick the correct channels
# - Notes: 
# -- Write them exactly the same as they were displayed in the previous code cell
# -- Write the channel names in the same order you want them compared.
#    For example, if you write pick_clean_chans = {'F3', 'P7'} and pick_test_chans = {'AF3', 'P7'}
#    The EQI will compare 'F3' vs 'AF3', and 'P7' vs 'P7'
pick_clean_chans = ['F3', 'C3', 'P7', 'O1']
pick_test_chans = ['AF3', 'FC5', 'P7', 'O1']

clean.pick(pick_clean_chans)
test.pick(pick_test_chans)

#%% Get data
# - Get numpy array data, and srate from MNE objects
clean_array = clean.get_data()
clean_srate = int(clean.info['sfreq'])

test_array = test.get_data()
test_srate = int(test.info['sfreq'])

#%% Compute EEG Quality Index
# - Set these values accordingly if srates are different between clean and test recordings
window_clean = int(clean_srate/2)   # Number of samples for EQI window, must be an integer
window_test = int(test_srate/2)     # Number of samples for EQI window, must be an integer
window = [window_clean, window_test]

slide_clean = 10    # Number of samples for sliding window, must be an integer
slide_test = 10     # Number of samples for sliding window, must be an integer
slide = [slide_clean, slide_test]

# - Disable the values above and enable these if srates are the same between clean and test
# window = 256    # Number of samples for EQI window, must be an integer
# slide = 5       # Number of samples for sliding window

[total, percent, eqi_mean] = eeg_quality_index.scoring(clean_eeg=clean_array, 
                                                        test_eeg=test_array,  
                                                        srate_clean=clean_srate,
                                                        srate_test=test_srate,
                                                        window=window,
                                                        slide=slide)

#%% Plot heatmap
sns.set_theme(style="white")

row_names = ['$SSAS_{1-50 Hz}$', '$SASS_{60 Hz}$', 'RMS', '$Grad_{max}$', 'ZCR', 'Kurtosis']
column_names = pick_clean_chans + ['Mean']

# Generate a large random dataset
percent_df = pd.DataFrame(data=np.concatenate((percent, np.mean(percent,1,keepdims=True)), axis=1), 
                            columns=column_names, index=row_names)

# Set up the matplotlib figure
f, ax = plt.subplots()

# Generate a custom diverging colormap
cmap1 = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap
sns.heatmap(percent_df, cmap=cmap1, vmax=100, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
ax.set_title('EEG Quality Index\nBaseline (raw) vs Clean (eyes open)')
plt.tight_layout()
plt.show()