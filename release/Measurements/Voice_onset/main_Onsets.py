# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 13:47:22 2022

@author: ariasvts

IMPORTANT: The acoustic, egg, and airflow signal MUST have the same sampling frequency

"""
import logging
import numpy as np
import scipy as sp
from scipy.stats import kurtosis
from scipy.signal import hilbert, gaussian
from Measurements.Voice_onset import VocalAttackTime,VocalRiseTime, VoiceOnsetTime


def VoiceOnset_Methods():
    DictMeth = {"None":0,
                'Vocal Rise Time':Acoustic_Onset} 
    return DictMeth

#======================================================================

def Envelope_Methods():
    DicMeth = {'Hilbert':hilb_tr,
               'Peak Amplitude':peakamp_method,
               'Root Mean Square':rms
               }
    return DicMeth
#======================================================================

def Acoustic_Onset(sig,fs,time_stamps,labels,method='VRT',kargs={}):
    
    Results = {}
    #-
    f=1
    for i in time_stamps:
        #-
        ini_time = i[0,0]
        ini_sample = int(fs*ini_time)
        #-
        end_time = i[0,1]
        end_sample = int(fs*end_time)
        #------------------------------
        # try:
        if method == 'VRT':          
            onsets = compute_vrt(sig[ini_sample:end_sample],fs,kargs)
            
        
        #Add the initial time of the segment that is being analyzed
        Results['Onset_'+labels[f-1]] = np.hstack([onsets['Start'],onsets['End']]).reshape(1,-1)
        Results['Envelope_'+labels[f-1]] = onsets['Envelope']
        Results['Features_'+labels[f-1]] = onsets['Features']
        # except:
            # logging.info('Error computing Voice onset on a segment: Start [seconds]: '+str(np.round(ini_time,3))+' End [seconds]: '+str(np.round(end_time,3)))
        #Onset index
        f+=1
    return Results


#======================================================================

def compute_vrt(sig,fs,kargs):
    """
    General method used to estimate the voice onset

    Parameters
    ----------
    sig : Acustic signal
    fs : sampling frequency
    kargs : Dictionary with the arguments used to compute the voice onset

    Returns
    -------
    Results : TYPE
        DESCRIPTION.

    """
    #Get general parameters
    segment = kargs['Segment']
    envmeth = kargs['Method']
    onset_point = kargs['Onset point']
    saturation_point = kargs['Saturation point']
    
    #Signal normalization
    sig = sig-np.mean(sig)
    sig = sig/np.max(sig)
    #----
    if segment>len(sig)/fs:
        segment = len(sig)/fs
    lseg = int(segment*fs)
    #-
    time_offset = 0.05#Add X ms at the beginning
    pre_vo = int(time_offset*fs)#To set new beginning of the signal
    #Pad zeros to ensure that the signal is sufficiently long for analysis.
    padd = np.random.normal(-0.001,0.001,pre_vo)
    sig = np.insert(sig,0,padd)
    padd = np.random.normal(-0.001,0.001,lseg)
    sig = np.insert(sig,len(sig),padd)
    #-
    #Get envelope
    emeth = Envelope_Methods()
    envelope = emeth[envmeth](sig,fs,kargs['Smooth factor'])
    #-
    #Use an initial voice onset point, in case there is too much silence at the beggining.
    vo = np.where(envelope>=(0.01*np.max(envelope)))[0][0]#voice initiation
    envelope = envelope[0:lseg+vo]
    #-
    envelope = envelope/np.max(np.abs(envelope))
    #-
    ti = np.where(envelope>=(onset_point*np.max(envelope)))[0][0]/fs#Select the first frame that satisfy the condition
    ti = ti - time_offset
    #In case the initial point is below the time offset
    # if ti<0:
    #     ti = 0
    ti = np.round(ti,3)
    #-
    tf = np.where(envelope>=(saturation_point*np.max(envelope)))[0][0]/fs#Select the first frame that satisfy the condition
    tf = tf - time_offset
    tf = np.round(tf,3)
    #-
    envelope = envelope[pre_vo:]
    #-
    X = feature_extraction(sig,envelope,fs,ti,tf)
    #-    
    Results = {'Envelope':envelope,
               'Start':ti,
               'End':tf,
               'Features':X}
    return Results

#======================================================================

def feature_extraction(sig,envelope,fs,ti,tf):
    #Normalize the points that form the right triangle
    x1 = ti/(len(envelope)/fs)#Normalized time
    x2 = tf/(len(envelope)/fs)
    # x1 = ti#Normalized time
    # x2 = tf
    y1 = envelope[int(ti*fs)]
    y2 = envelope[int(tf*fs)]
    # #Compute the vertices
    # hyp = euc_dist(x1,y1,x2,y2)#Hypotenuse
    # adj = euc_dist(x1,y1,x2,y1)#Adjacent
    #Angle between hypothenuse and adjacent
    # theta = np.arccos(adj/hyp)
    theta = np.arctan((y2-y1)/(x2-x1))
    # Phase in degrees
    slope = int(theta*180/np.pi)
    #-
    # #Area triangle rise time
    # opp = np.sqrt((hyp**2)-(adj**2))
    # area = 0.5*(adj*opp)
    
    #Kurtosis
    kur = np.round(kurtosis(sig[int(ti*fs):int(tf*fs)]),4)
    
    
    X = {'Slope (Degrees)':slope,
         'Kurtosis':kur}
    return X

#*****************************************************************************
def hilb_tr(signal,fs,factor=2):
    """
    Apply hilbert transform over the signal to get the envelope
    
    Parameters
    ----------
    signal : Acoustic signal
    fs : sampling frequency
    smooth_window : The smooth window controls the size of the Gaussian window 
    that is convolved with the envelope for smoothing. Typical values are in the
    range of 10ms <= smooth_window <= 100ms

    Returns
    -------
    amplitude_envelope : Array with amplitude values form the envelope

    """
    #Hilbert Transform
    analytic_signal = hilbert(signal)
    #Amplitude Envelope
    amplitude_envelope = np.abs(analytic_signal)
    #-
    # yreal = analytic_signal.real
    # yimag = analytic_signal.imag
    # phase_envelope = np.arctan(yimag.copy(),yreal.copy())
    #-
    #Temporal Fine Structure
    # tfs = analytic_signal.imag/amplitude_envelope
    #-
    smooth_window = 0.04*factor
    #Convolve amplitude evelope with Gaussian window
    if smooth_window!=0:
        amplitude_envelope = smooth_curve(amplitude_envelope,fs,smooth_window)
        # phase_envelope = smooth_curve(phase_envelope,fs,smooth_window)
    return amplitude_envelope

#*****************************************************************************

def smooth_curve(x,fs,glen = 0.01):
    #Gaussian Window
    gauslen = int(fs*glen)
    window = gaussian(gauslen, std=int(gauslen*0.05))
    #Convolve signal for smmothing
    smooth_x = x.copy()
    smooth_x = sp.convolve(smooth_x,window)
    smooth_x= smooth_x/np.max(smooth_x)
    ini = int(gauslen/2)
    fin = len(smooth_x)-ini
    x = smooth_x[ini:fin]
    return x

#****************************************************************************

def peakamp_method(signal,fs,factor=6):
        """
        Computes the envelope by interpolating the amplitude peaks of the FILTERED
        signal. 
        
        Ideally, a bandpass filter with cut-off frequency between 75Hz and (F0+10)Hz
        is used, where F0 is the average fundamental frequency of the signal.
        #----------       
        """
        peak_dst = int(4**factor)
        #Get envelope by bspline interpolation of amplitude peak points
        peaks, _ = sp.signal.find_peaks(signal, distance=peak_dst,height=0)#Indexes
        points = signal[peaks]#Amplitude values
        
        spl = sp.interpolate.splrep(peaks, points)
        xvals = np.linspace(peaks[0],peaks[-1],len(signal))
        envelope = sp.interpolate.splev(xvals, spl)

        return envelope

#****************************************************************************

def rms(sig,fs,factor=2):
    """
    Sound Pressure Level as in:
        Švec JG, Granqvist S. Tutorial and Guidelines on Measurement of Sound 
        Pressure Level in Voice and Speech. Journal of Speech, Language, and Hearing Research. 
        2018 Mar 15;61(3):441-461. doi: 10.1044/2017_JSLHR-S-17-0095. PMID: 29450495.
        
    SPL = 20*log10(p/p0)
    
    20xlog refers to a root-power quantity e.g., volts, sound pressure, current...
    
    Intensity in dBs:
        ene = 10*log10(sum(x^2)/N)
    
    10xlog refers to a power quantity, i.e. quantities directly proportional to power
    
    x: speech signal
    N: lenght of x
    p = RMS value of x
    p0 = 20uPA = 0.00002 Hearing threshold
    """
    step_time = 0.001#The size of the step size (time resolution) is fixed to 1ms
    #win_time = 0.025#The size of the frames is fixed to 25ms
    #The window length (window size of frames) is controlled with "factor"
    if factor == 0:
        win_time = 0.001 #1ms time resolution
    else:
        win_time = 0.02*factor
    #Set a threshold based on the energy of the signal
    if len(sig)>3*int(win_time*fs):
        frames = extract_windows(sig,int(win_time*fs),int(step_time*fs))
    else:
        frames = list([sig])
    EnergySignal = []
    # p0 = 2*(10**-5)#Hearing threshold at SLP 0dB
    # eps = np.finfo(np.float32).eps#To avoid errors
    for x in frames:
        #Sound Pressure Level (dBs)
        p = np.sqrt(np.sum((x)**2)/(len(x)))
        # Lp = 20*np.log10((p/p0)+eps)
        # Lp = 10*np.log10(p+eps)
        EnergySignal.append(p)
    EnergySignal = np.asarray(EnergySignal)
    
    #Interpolating the RMS values to match the length of the signal
    start = 0
    stop = EnergySignal[-1]
    N = len(EnergySignal)
    samples = EnergySignal.copy()#Samples to interpolate'
    Nsamples = len(sig)#Number of points create from the inteprolation
    #-
    index = np.linspace(start,stop,N)
    pl = sp.interpolate.splrep(index, samples)
    #-
    xvals = np.linspace(start,stop,Nsamples)
    envelope = sp.interpolate.splev(xvals, pl)
    
    return envelope

#****************************************************************************

def extract_windows(signal, size, step,windowing = True):
    # make sure we have a mono signal
    assert(signal.ndim == 1)
    
#    # subtract DC (also converting to floating point)
#    signal = signal - signal.mean()
    
    n_frames = int((len(signal) - size) / step)
    
    # extract frames
    windows = [signal[i * step : i * step + size] 
               for i in range(n_frames)]
    
    # stack (each row is a window)
    windows = np.vstack(windows)
    if windowing == True:
        windows = windows*np.hanning(size)
    return windows

#****************************************************************************

def euc_dist(x1,y1,x2,y2):
    return np.sqrt(((x2-x1)**2)+((y2-y1)**2))

#======================================================================        
        
def get_segments(fs,segments):
    """
    Get time stamps of the speech/voice segments after computing the energy contour

    Parameters
    ----------
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
        seg_time.append([tini,tend])
        
    return seg_time
