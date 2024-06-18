# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 12:42:36 2021

@author: ariasvts
"""
import adi
import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.io.wavfile import read

def read_labchart(file_name,res_flag = True,rs = 16000,channelname=None):
    """
    Read Labchart (.adicht) data
    
    All data is resampled to 16kHz for analysis. The original files 
    are not modified.
    
    Parameters
    ----------
    file_name : string
        Labchart file to be loaded
        
    rs : If res_flag is True, the labchart data will be resampled at the specified sampling frequency (default: 16kHz)
    
    res_flag : Bool
        If True, the data will be resampled to the sampling frequency determine by "rs"
    
    channelname: is used to exteract an specific channel from the file. If None, then every possible channel will be extracted
    Returns
    -------
    data : TYPE
        DESCRIPTION.

    """
    #Read labchart
    file_data = adi.read_file(file_name)
    
    
    # #Get all channels
    # list_channels = []
    # for c in file_data.channels:
    #     list_channels.append(c.name)
    
    #Look for acoustic, egg, and airflow data
    list_channels = []
    if channelname==None:
        for channel in file_data.channels:
            # if (channel.name == 'Acoustic mic')|(channel.name == 'EGG')|(channel.name == 'Airflow-Input A'):
                list_channels.append(channel.name)
    else:
        list_channels.append(channelname)
    #Make sure that at least one of the channels is in the labchart
    if len(list_channels)>0:
        #Get the acoustic, egg, and airflow data
        data = get_channel_data(file_data, list_channels)
        
        #Number of records in labchart data
        for r in list_channels:
            n = data[list_channels[0]].n_records
            if n!=0:
                n_records = n
                break
        
        #Read all possible records stored in the labchart file
        data_total = {} # Store all possible records
        total_channels = []
        for irec in range(n_records):
            data_chan = {} # Store all channel data per record
            for channel in list_channels:
                try:
                    #Read signal and sampling frequency
                    fs = data[channel].fs[irec]
                    sig = data[channel].get_data(irec+1)
                    
                    #DC removal and re-scaling
                    sig = sig-np.mean(sig)
                    # sig = sig/np.max(np.abs(sig))
                    
                    #Should the data be resampled?
                    if res_flag:
                        #Resample data if necessary
                        if fs!=rs:
                            sig = resample_data(sig,fs,rs)
                            fs = rs
                                
                    #Store information from signal
                    data_sig = {'signal':sig,
                                'fs':fs}
                    #Store data from an specific record
                    data_chan[channel] = data_sig
                    total_channels.append(channel)
                except: None
                
            data_total['record_'+str(irec+1)] = data_chan
            
    #If there is no data with the selected channels, then return 0
    else:
        data_total = 0
    return data_total,total_channels

#=========================================================================         

def get_channel_data(file_data,list_channels):
    """
    Get the specified data form the labchart file: acoustic, egg, airflow,...
    """
    data = {}
    for channel in file_data.channels:
        for selected in list_channels:
            if channel.name == selected:
                data[selected] = channel
    return data

#=========================================================================         

def read_wav(file_name,res_flag = True,rs = 16000):
    """
    Read wav files and perform resampling

    Parameters
    ----------
    file_name : TYPE
        DESCRIPTION.
    res_flag : TYPE, optional
        DESCRIPTION. The default is True.
    rs : TYPE, optional
        DESCRIPTION. The default is 16000.

    Returns
    -------
    data_total : TYPE
        DESCRIPTION.
    channel: Return the name of the channel as a list. This is to keep the structure
              of the software

    """
    data_total = {}
    data_chan = {}
    #Does the file even exist?
    try:
        fs,signal = read(file_name)
        nchannels = len(signal.shape)#Number of channels
        #Get the channels from audio
        if nchannels>1:
            list_sig = []
            channel = []
            for ch in range(nchannels):
                list_sig.append(signal[:,ch])
                channel.append('channel_'+str(ch+1))
        else:
            list_sig = [signal]
            channel = ['channel_1']
        
        for i in range(len(list_sig)):
            sig = list_sig[i]
            #Re-scaling
            sig = sig - np.mean(sig)
            sig = sig/np.max(np.abs(sig))
            
            #Should the data be resampled?
            if res_flag:
                #Resample data if necessary
                if fs!=rs:
                    sig = resample_data(sig,fs,rs)
                    
            #Store information from signal. It is performed in a similar way to the 
            #adicht files in order to keep the same structure
            data_sig = {'signal':sig,
                        'fs':rs}
            data_chan[channel[i]] = data_sig
        data_total['record_1'] = data_chan
    except:
        data_total = 0
    return data_total,channel

#=========================================================================         

def read_GAT_GAW(file_name):
    """
    Read CSV files from the Glottal Analysis Tool (GAT)

    Parameters
    ----------
    file_name : TYPE
        DESCRIPTION.

    Returns
    -------
    data_total : TYPE
        DESCRIPTION.
    channel: Return the name of the channel as a list. This is to keep the structure
              of the software

    """
    data_total = {}
    data_chan = {}
    
    list_channels = ['Total GAW','Left GAW','Right GAW']
    #-
    try:    
        data = pd.read_csv(file_name,sep=';')
        
        columns = list(data.columns)
        
        #Get sampling frequency
        try:
            ts = data['[Time(s)]'][0].replace(',','.')#German format put commas instead of dots for decimal points
        except:
            ts = data['[Time(s)]'][0]
        fs = int(1/np.float(ts))
        
        
        for i in range(2,5):
            #Replace commas ',' by dots '.'.
            try:
                sig = data[columns[i]].str.replace(',', '.', regex=True) #Weird German decimal point format is weird
                sig = np.asarray(pd.to_numeric(sig))
            except:
                sig = np.asarray(pd.to_numeric(data[columns[i]]))
               
              
            sig = sig/np.max(np.abs(sig))
            data_sig = {'signal':sig,
                        'fs':fs}
            data_chan[list_channels[i-2]] = data_sig
        data_total['record_1'] = data_chan
    except:
        data_total = 0
    return data_total,list_channels

#=========================================================================         

def read_GAT_Trajectory(file_name,sel_traj,fs=4000):
    """
    
    Read PVG file (.csv) from the Glottal Analysis Tool (GAT)

    Parameters
    ----------
    file_name : TYPE
        DESCRIPTION.
    sel_traj : Used to select the trajectory to be read from the file
    fs: Sampling frequency. GAT 2020 does not include this information in the csv file

    Returns
    -------
    data_total : TYPE
        DESCRIPTION.
    list_channels : TYPE
        DESCRIPTION.

    """
    data_total = {}
    data_chan = {}
    list_channels = {}
    
    
    try:
        list_channels = ['Left Trajectory','Right Trajectory']
        #Read GAT-PVG csv file
        data = pd.read_csv(file_name,sep=';')
        
        #Check that a CSV PVG file (GAT2020) is being used.
        #This will raise an error if the column is not in the CSV file
        #I know is stupid, but I do not give a fuck anymore.
        dummy = data['[Position]'][0]
        
        #Number of rows
        Nrows = data.shape[0]-1
        index_Left = int((Nrows/2)-(Nrows/2)*sel_traj)
        index_Right = int(Nrows-index_Left)
        #-
        Left_data = data.loc[index_Left,:]
        Right_data = data.loc[index_Right,:]
        try:
            Left_data = Left_data.str.replace(',', '.', regex=True)
            Left_data = Left_data.to_numpy()
            Right_data = Right_data.str.replace(',', '.', regex=True)
            Right_data = Right_data.to_numpy()
        except: 
            Left_data = Left_data.to_numpy()
            Right_data = Right_data.to_numpy()
        #-------------------------------------------------------------
        Left_data = Left_data[1:].astype(float)
        Left_data[np.isnan(Left_data)] = 0
        # Left_data[Left_data<0] = 0
        #-
        sig = Left_data/np.max(np.abs(Left_data))
        data_sig = {'signal':sig,
                    'fs':fs}
        data_chan['Left Trajectory'] = data_sig
        #----------------------------------------------------------
        Right_data = Right_data[1:].astype(float)
        Right_data[np.isnan(Right_data)] = 0
        # Right_data[Right_data<0] = 0
        #-
        sig = Right_data/np.max(np.abs(Right_data))
        data_sig = {'signal':Right_data/np.max(np.abs(Right_data)),
                    'fs':fs}
        data_chan['Right Trajectory'] = data_sig
        #----------------------------------------------------------
        data_total['record_1'] = data_chan
    except:
            data_total = 0
    return data_total,list_channels


#=========================================================================    

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
    
    #Re-scaling
    sig = sig - np.mean(sig)
    sig = sig/np.max(np.abs(sig))
    
    return sig