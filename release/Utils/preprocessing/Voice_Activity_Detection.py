# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 15:11:40 2021

@author: ariasvts
"""

#-

import numpy as np
import scipy as sp
from scipy.signal import gaussian


def VAD_methods():
    DictMeth = {"None":0,
                "Manual":0,#This is for the VOAT.exe GUI
                "Energy":eVAD}
    return DictMeth
    
#======================================================================

def eVAD(sig,fs,win=0.015,step=0.01,VAD_thr=None):
    """
    Energy-based Voice Activity Detection
    
    Parameters
    ----------
    sig : Recordings to be analyzed
    fs : Sampling frequency
    win : Size of the analysis window. Default is 15ms
    step : Step size to frame the signal. Default is 10ms
    
    Returns
    -------
    seg_time : List with time stamps where there are speech/voice segments
    """    
    #Normalize signal
    sig = sig-np.mean(sig)
    sig /=np.max(np.abs(sig))
    
    lsig = len(sig)
    
    #-------------------------------------------------------------------- 
    #Add silence based on minimum signal energy
    #Set min threshold base on the energy and minimum amplitude of the signal
    e,frames = compute_energy(sig,fs,win,step,False)
    idx_min = np.where(e==np.min(e))
    thr = np.min(frames[idx_min])
    #---
    #Add silence
    ext_sil = int(fs)
    esil = int((ext_sil/2)/fs/step)
    new_sig = np.random.randn(lsig+ext_sil)*thr
    new_sig[int(ext_sil/2):lsig+int(ext_sil/2)] = sig
    sig = new_sig
    #-------------------------------------------------------------------- 

    #-------------------------------------------------------------------- 
    #------Compute energy contour-------------------------------------
    e,_ = compute_energy(sig,fs,win,step,True)
    e = e-np.mean(e)
    #Smooth energy contour to remove small energy variations
    gauslen = int(fs*0.01)
    window = gaussian(gauslen, std=int(gauslen*0.05))
    #Convolve signal with Gaussian window for smoothing
    smooth_env = e.copy()
    smooth_env = sp.convolve(e,window)
    smooth_env = smooth_env/np.max(smooth_env)
    #Add padding. The convolution "removes" small segments from start and end 
    #of the signal
    ini = int(gauslen/2)
    fin = len(smooth_env)-ini
    e = smooth_env[ini:fin]
    e = e/np.max(np.abs(e))
    #Take out the silence
    e = e[esil:int(lsig/fs/step)+esil]
    #Compute the minimum threshold used to decide if a frame is speech or silence
    if VAD_thr is None:
        VAD_thr = np.median(e[e<0])/2
    else:
        #If selected based on the amplitude, then
        #it should be converted to the normalized energy value
        ethr,_ = compute_energy(sig,fs,win,step,True)
        VAD_thr = VAD_thr-np.mean(e)
    #Label the silence and speech/voice segments
    cont_sil = np.zeros(lsig)
    cont_vad = np.zeros(lsig)
    itime = 0
    etime = int(win*fs)
    for i in range(len(e)):
        if e[i]<=VAD_thr:
            cont_sil[itime:etime] = 1
        else:
            cont_vad[itime:etime] = 1
            
        itime = i*int(step*fs)
        etime = itime+int(win*fs)
        
    #Remove silence added at the begining
    sig = sig[int(ext_sil/2):lsig+int(ext_sil/2)]
    #Verify that there is at least one potential speech segment
    if np.sum(cont_sil)!=0:
        #Voice
        time_vad = get_segments(sig,fs,cont_vad)
        if len(time_vad)!=0:
            time_vad = np.vstack(time_vad)
        else:
            time_vad = []
    else:
        time_vad = []
    return time_vad,VAD_thr

#======================================================================

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
    frames = extract_windows(sig,int(win*fs),int(step*fs),windowing)
    for seg in frames:
        e.append(10*np.log10(np.sum(np.absolute(seg)**2)/len(seg)))
    e = np.asarray(e)
    return e,frames

#======================================================================

def extract_windows(signal, size, step,windowing=True):
    # make sure we have a mono signal
    assert(signal.ndim == 1)
    
    n_frames = int((len(signal) - size) / step)
    
    # extract frames
    windows = [signal[i * step : i * step + size] 
               for i in range(n_frames)]

    # stack (each row is a window)
    windows = np.vstack(windows)
    
    #Perform windowing?
    if windowing==True:
        windows*=np.hanning(int(size))
    return windows

#======================================================================

def get_segments(sig,fs,segments):
    """
    Get time stamps of the speech/voice segments after computing the energy contour

    Parameters
    ----------
    sig : Recordings to be analyzed
    fs : Sampling frequency
    segments : array with labels of speech sounds

    Returns
    -------
    seg_time : List with time stamps

    """
    segments[0] = 0
    segments[-1:] = 0
    yp = segments.copy()
    ydf = np.diff(yp)
    lim_end = np.where(ydf==-1)[0]+1
    lim_ini = np.where(ydf==1)[0]+1
    #Silence segments
    seg_time = []#Time stamps
    for idx in range(len(lim_ini)):
        #------------------------------------
        tini = lim_ini[idx]/fs
        tend = lim_end[idx]/fs
        #Is the segment longer than 200ms?
        if (tend-tini)>0.1:
            #Is the maximum amplitude (in the segment) higher than 0.1?
            if np.max(np.abs(sig[lim_ini[idx]:lim_end[idx]]))>0.1:
                seg_time.append([tini,tend])
        
    return seg_time
    
