"""
    Import data

"""

import pandas as pd
import numpy as np

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