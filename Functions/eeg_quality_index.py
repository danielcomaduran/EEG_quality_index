"""
    EEG Quality Index
    -----------------
        These functions are used to quantify the EEG quality index presented in Flicking et al. 2019.
"""

## Import libraries
import numpy as np
import scipy.fft as fft
import scipy.stats as stats

def scoring(clean_eeg, test_eeg, sliding=True, window=None, slide=None):
    """
        This function computes the EEG Quality index for both the clean_eeg and the test_eeg and returns
        a scoring matrix for the following variables: 
        - Average single-sided amplitude spectrum (1-50 Hz)
        - Line noise single-sided amplitude spectrum (59-61 Hz)
        - RMS amplitude
        - Maximum gradient
        - Zero-crossing rate
        - Kurtosis

        The scores for each variable are determined with the following table
        - EQI_test <= 1 stdev EQI_clean = 0
        - 

    """

def eqi(eeg, srate, sliding=True, window=None, slide=None):
    """
        This function calculates the EEG Quality index and returns all metrics in a 2D array

        Parameters
        ----------
            eeg: array_like
                EEG array for which the EQI will be calculated
            srate: int
                Sampling rate [Hz]
            sliding: bool (optional)
                Boolean to calculate sliding window
                If true, EQI will be calculated for each window
                If false, EQI will be calculated for the whole EEG data
            window: int (optional)
                Number of samples to calculate the sliding window
                Required if sliding == True
            slide: int (optional)
                Number of samples to slide the sliding window for
                Required if sliding == True
        
        Returns
        -------
            eeg_eqi: array_like
                EEG quality index values. 

    """

    ## Make sure that data is in a column format
    eeg_size = eeg.shape
    if eeg_size[0] > eeg_size[1]:
        eeg = eeg.T

    ## Compute sliding window if needed
    if sliding:
        eeg_windowed = sliding_window(eeg, window, slide)

    ## Compute single-sided amplitude spectrum
    [ssas, f] = single_amplitude_spectrum(eeg_windowed, srate, n=srate) # Compute FFT for a 1 sec window
    ssas_size = ssas.shape

    # - Compute average
    f_start = 1     # Frequency start for average [Hz]
    f_end = 50      # Frequency end for average [Hz]
    f_mask = np.expand_dims((f>=f_start) & (f<=f_end), axis=0)
    f_mask_tensor = np.expand_dims(f_mask.repeat(ssas_size[0], axis=0), axis=2).repeat(ssas_size[2], axis=2)
    
    mean_ssas = np.mean(ssas, axis=1, where=f_mask_tensor)    # Mean Single Sided Amplitude Spectrum
    
    ## Line noise average
    f_start = 59    # Frequency start for average [Hz]
    f_end = 61      # Frequency end for average [Hz]
    f_mask = np.expand_dims((f>=f_start) & (f<=f_end), axis=0)
    f_mask_tensor = np.expand_dims(f_mask.repeat(ssas_size[0], axis=0), axis=2).repeat(ssas_size[2], axis=2)

    line_ssas = np.mean(ssas, axis=1, where=f_mask_tensor)    # Line noiseMean Single Sided Amplitude Spectrum

    ## RMS Amplitude
    rms = np.sqrt(np.mean(eeg_windowed**2, axis=1)) 

    ## Maximum gradient
    max_grad = (np.diff(eeg_windowed, axis=1)).max(axis=1) 

    ## Zero-crossing rate
    eeg_zcr = zcr(eeg_windowed)

    ## Kurtosis
    eeg_kurtosis = stats.kurtosis(eeg_windowed, axis=1)

def sliding_window(data, window, slide):
    """
        This function calculates a sliding window across the longest dimension of a 2D array and returns a 
        3D tensor where the 3rd dimension are the windows

    Parameters
    ----------
        data: 2D array_like
            Data to be divided into sliding windows
        window: int
            Number of samples for each window
        slide: int
            Number of samples to slide each window
    Returns
    -------
        data_windowed: 3D array_like   
    """
    
    # Important size values
    shape = data.shape
    n_chans = shape[0]                      # Number of channels [n]
    data_length = shape[1]                  # Input data length [n]                 
    max_n_windows = data_length-window+1    # Max number of windows required [n]

    # Create matrix for indices of the sliding window
    window_idy = np.arange(0,max_n_windows,slide).reshape(-1,1).repeat(window, axis=1)  # Index for columns
    window_idx = np.arange(window).reshape(1,-1).repeat(np.size(window_idy,0), axis=0)  # Index for rows
    window_mat = window_idx + window_idy                                                # Matrix with index of windows
    window_shape = window_mat.shape

    # Preallocate output data
    data_windowed = np.zeros((n_chans, window_shape[1], window_shape[0]))

    # Fill each window with the right index values
    for w in range(window_shape[0]):
        data_windowed[:,:,w] = data[:,window_mat[w,:].astype(int)]
    
    return data_windowed

def single_amplitude_spectrum(eeg, srate, n):
    """
        This function calculates the single-sided amplitude spectrum in a column-wise fashion

        Parameters
        ----------
            eeg: array_like
                EEG data to calculate the single-sided amplitude spectrum
            srate: int
                Sampling rate [Hz]
            n: int
                Number of points to compute the fft 

        Returns
        -------
            single_fft: array_like
                Single-sided amplitude spectrum magnitude
            f: array_like
                Frequency vector corresponding to the single-sided amplitude spectrum magnitudes
    """

    # Calculate single sided FFT and frequency vector
    single_fft = np.abs(fft.rfft(eeg, n=n, axis=1, workers=-1))

    # Create frequency vector
    size_fft = np.shape(single_fft)
    f = srate * np.linspace(0, n/2, size_fft[1]) / n    # Frequency vector [Hz]

    return single_fft, f

def zcr(data):
    """
        Compute the zero-crossing rate (ZCR) of the input data along the columns

    Parameters
    ----------
        data: array_like
            Data to compute the ZCR

    Returns
    -------
        data_zcr: array_like
            ZCR result

    Notes
    -----
        - ZCR will have one less dimension than the input data
    """

    data_shape = data.shape

    # Make sure data is in a row matrix
    if data_shape[0] > data_shape[1]:
        data = data.T

    data_zcr = np.mean(np.diff(np.sign(data), axis=1), axis=1)

    return data_zcr