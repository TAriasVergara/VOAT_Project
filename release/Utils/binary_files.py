# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 11:16:32 2021

@author: ariasvts
"""

import os
import struct
import numpy as np

def save_binary(data,fs,filename,ext='.bin'):
    """
    Save a signal as a binary file

    Parameters
    ----------
    data : array
    fs : sampling frequency
    filename : file name
    ext : extension of the file
        DESCRIPTION. The default is '.bin'.

    Returns
    -------
    None.

    """
    file = open(filename+ext, "wb")
    file.write(bytes(fs))
    file.write(bytes(len(data)))
    file.write(bytes(data))
    file.close()
    
    
def load_binary(filename):
    file = open(filename, "rb")
    Nbytes = 8
    fs = int(struct.unpack('d',file.read(Nbytes))[0])
    lsig = int(struct.unpack('d',file.read(Nbytes))[0])
    data = []
    for i in lsig:
        data.append(int(struct.unpack('d',file.read(Nbytes))[0]))
    data = np.asarray(data)
    return data,fs