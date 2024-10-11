"""
    Import data

"""

import pandas as pd
import numpy as np
import pyxdf
import mne

scaling_factors = {
    "volts": 1,
    "millivolts": 1e-3,
    "microvolts": 1e-6,
}

def import_openBCI(file:str):
    """
        Import openBCI

        Parameters
        ----------
            file: str
                Complete file path name of the .TXT file to import


        Returns

    """

    eeg = pd.read_csv(file, header=4, usecols=range(1,17,1))

    return eeg.to_numpy()

def xdf_to_mne(file:str) -> mne.io.Raw:
    """
        Import `xdf` file and returns an MNE raw object

        Parameters
        ----------
            file: str
                Path of the .XDF file to import


        Returns
        -------
            raw: mne.io.Raw
                MNE raw object

    """

    # Load the xdf file
    streams = pyxdf.load_xdf(file)[0]

    # Get the EEG stream
    for stream in streams:
        if stream["info"]["type"][0] == "EEG":
            eeg_stream = stream
    
    # Get the channel names and indices
    channels = eeg_stream["info"]["desc"][0]["channels"][0]["channel"]
    ch_indices = []
    ch_scaling = []
    ch_names = []
    for (c,channel) in enumerate(channels):
        if channel["type"][0] == "EEG":
            ch_names.append(channel["label"][0])
            ch_scaling.append(scaling_factors[channel["unit"][0]])
            ch_indices.append(c)

    # Get the EEG data
    eeg_data = eeg_stream["time_series"][:,ch_indices].T
    eeg_data = eeg_data * np.array(ch_scaling)[:, np.newaxis]

    # Create the MNE info object
    sfreq = eeg_stream["info"]["nominal_srate"][0]
    info = mne.create_info(ch_names, sfreq=sfreq, ch_types="eeg")

    # Create MNE Raw object
    raw = mne.io.RawArray(eeg_data, info)

    return raw