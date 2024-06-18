# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 09:17:05 2021

@author: ariasvts
"""

import numpy as np
import scipy as sp
from scipy.signal import hilbert, gaussian


def filtering(sig,fs,cutoff,ftype,order=4):    
    """
    Apply a Butterworth filter.

    Parameters
    ----------
    sig : Signal to be filtered
    fs : Sampling frequency of the signal
    cutoff : Cut-off frequency in hertz. For a bandpass filter, the cutoff filter 
            must be a list. Examples:
            bandpass 75Hz-300Hz -> cutoff = [75,300]
            highpass 420Hz -> cutoff = 420
    ftype : bandpass,highpass, or lowpass
    order : Filter order
        DESCRIPTION. The default is 4.

    Returns
    -------
    filtered : filtered signal

    """
    # Convert frequency to Hertz
    nyq = 0.5*fs
    if type(cutoff)==list:
        if cutoff[0]<cutoff[1]:
            fLow = cutoff[0]
            fHigh = cutoff[1]
        else:
            fLow = cutoff[1]
            fHigh = cutoff[0]
        #Apply bandpass/stopband
        b,a = sp.signal.butter(order, [fLow / nyq, fHigh / nyq], btype=ftype, analog=False)
    else:
        #Apply bandpass/stopband
        b,a = sp.signal.butter(order, cutoff / nyq, btype=ftype, analog=False)
    
    #Apply filter
    filtered = sp.signal.filtfilt(b, a, sig)
    #Re-scale filter between -1,+1
    filtered = filtered-np.mean(filtered)
    filtered = filtered/np.max(np.abs(filtered))
    return filtered

def get_spectrum(X,fs,win_time=0.025,step_time=0.01,n_padded=1024):
    """
    Compute log-power spectrum of a signal

    Parameters
    ----------
    X : Can be an array or a matrix. If X is an array, then it is assume that this
        is the signal. If X is a matrix, then it is assume that is the framed version
        of the signal
    fs : sampling frequency
    win_time : Duration of the analysis signal measured in seconds. The default is 0.025.
    step_time : Step size measured in seconds. The default is 0.01.
    n_padded : Resolution of the spectrum. The default is 1024.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    #Convert from time to number of samples
    win_size = int(fs * win_time)
    step_size = int(fs * step_time)
    
    #If the signal is an array, then use windowing
    if X.ndim==1:
        X = extract_windows(X, win_size, step_size)
    
    # apply hanning window
    X *= np.hanning(win_size)

    # Fourier transform
    Y = np.fft.fft(X, n=n_padded)
    # non-redundant part
    m = int(n_padded / 2) + 1
    Y = Y[:, :m]
    
    return np.log(np.abs(Y) ** 2+np.finfo(float).eps)

#---------------------------------------------------------------------------

def extract_windows(signal, win_size, step_size):
    """
    Frame the signal into windows of size "win" taken every "step" from 
    the singal

    Parameters
    ----------
    signal : Signal to be framed
    win_size : Size of the analysis window measured in number of samples
    step_size : Size of the step measured in number of samples
    Returns
    -------
    None.

    """
    # make sure we have a mono signal
    assert(signal.ndim == 1)
    
#    # subtract DC (also converting to floating point)
#    signal = signal - signal.mean()
    
    n_frames = int((len(signal) - win_size) / step_size)
    
    # extract frames
    windows = [signal[i * step_size : i * step_size + win_size] 
               for i in range(n_frames)]
    
    # stack (each row is a window)
    return np.vstack(windows)

#---------------------------------------------------------------------------

def smooth_energy_envelope(sig,fs,win=0.015,step=0.01):
    e,_ = compute_energy(sig,fs,win,step,True)
    e = e-np.mean(e)
    #Smooth energy contour to remove small energy variations
    gauslen = int(fs*0.01)
    window = gaussian(gauslen, std=int(gauslen*0.05))
    #Convolve signal with Gaussian window for smoothing
    smooth_env = e.copy()
    smooth_env = sp.convolve(e,window)
    smooth_env = smooth_env/np.max(smooth_env)
    return smooth_env

#---------------------------------------------------------------------------

def compute_energy(sig,fs,win,step,windowing):
    """
    Compute energy contour using short-time frames

    Parameters
    ----------
    sig : Signal to be analyzed
    fs : Sampling frequency of the signal
    win : Size of the analysis windows in seconds
    step : Step size in seconds
    windowing : If True, perform hanning windowing of the signal

    Returns
    -------
    e : energy contour computed frame-wise
    frames: framed signal

    """
    e = []#energy in dB
    frames = extract_windows(sig,int(win*fs),int(step*fs))
    frames*=np.hanning(int(win*fs))
    for seg in frames:
        e.append(10*np.log10(np.sum(np.absolute(seg)**2)/len(seg)))
    e = np.asarray(e)
    return e,frames

#---------------------------------------------------------------------------

def hilb_tr(signal,fs,smooth=True,glen = 0.01):
    """
    Apply hilbert transform over the signal to get
    the envelop and time fine structure
    
    If smooth true, then the amplitude envelope is smoothed with a gaussian window
    """
    #Hilbert Transform
    analytic_signal = hilbert(signal)
    #Amplitude Envelope
    amplitude_envelope = np.abs(analytic_signal)
    
    #Temporal Fine Structure
    tfs = analytic_signal.imag/amplitude_envelope
    
    #Convolve amplitude evelope with Gaussian window
    if smooth==True:
        #Gaussian Window
        gauslen = int(fs*glen)
        window = gaussian(gauslen, std=int(gauslen*0.05))
        #Convolve signal for smmothing
        smooth_env = amplitude_envelope.copy()
        smooth_env = sp.convolve(amplitude_envelope,window)
        smooth_env = smooth_env/np.max(smooth_env)
        ini = int(gauslen/2)
        fin = len(smooth_env)-ini
        amplitude_envelope = smooth_env[ini:fin]
        amplitude_envelope = np.insert(amplitude_envelope, 0, 0)
    return amplitude_envelope,tfs