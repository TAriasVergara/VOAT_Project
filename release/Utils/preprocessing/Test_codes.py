# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 09:52:53 2021

@author: ariasvts
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from scipy.io.wavfile import read
import scipy.signal as signal
import Voice_Activity_Detection as VAD


#------------------------------------------------------------------------------
def make_error_boxes(ax, xdata, ydata, xerror, yerror, facecolor='r',
                     edgecolor='None', alpha=0.4):

    # Loop over data points; create box from errors at each point
    errorboxes = [Rectangle((x, y), xe, ye)
                  for x, y, xe, ye in zip(xdata, ydata, xerror.T, yerror.T)]

    # Create patch collection with specified colour/alpha
    pc = PatchCollection(errorboxes, facecolor=facecolor, alpha=alpha,
                         edgecolor=edgecolor)

    # Add collection to axes
    ax.add_collection(pc)
    
#------------------------------------------------------------------------------
    
def resample_data(sig,fs,rs):
    """
    Resample signal

    Parameters
    ----------
    sig : signal to be re-sampled
    fs : Current sampling frequency of the signal
    rs :New sampling frequency

    Returns
    -------
    sig : resampled signal

    """
    num = int((len(sig)/fs)*rs)
    sig = signal.resample(sig,num)
    return sig

#------------------------------------------------------------------------------

path_file = 'E:/Projects/VoiceOnset_Sydney/Data_Sydney/Audios'
filename = path_file+'/S01_VOMS_280417_0.wav'

#Signal
fs,sig = read(filename)

#Resample data
rs = 16000
sig = resample_data(sig,fs,rs)
fs = rs

#Normalization
sig = sig-np.mean(sig)
sig = sig/np.max(np.abs(sig))

#VAD
times = VAD.eVAD(sig,fs)
# times = np.vstack(times)
# times = times*fs

#---------------------Plotting-------------------------------------------
fig,ax = plt.subplots(1)
t = np.arange(0,len(sig)/fs,1/fs)
ax.plot(t,sig,'k')
#-
# ax.twinx()
#-
# te = np.arange(0,len(e)*0.01,0.01)
# ax.plot(te,e,'r')
ax.set_xlim([0,t[-1:]])
#-
#Rectangle
xdata = times[:,0]#Anchor point x
ydata = np.zeros(times.shape[0])#Anchor point y
xerr = times[:,1]-times[:,0]#Width of the box
yerr = np.ones(len(ydata))*2#Hihg of the box
make_error_boxes(ax,xdata,ydata,xerr,yerr)
#------------------------------------------------------------------------
