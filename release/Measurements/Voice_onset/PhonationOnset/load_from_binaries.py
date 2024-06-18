from PhonationOnset.adinstruments_sdk_python import adi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from scipy import interpolate
from scipy.interpolate import interp1d
from os.path import isfile, join
from os import walk
from PhonationOnset.pyAudioAnalysis import audioSegmentation as aS
import flammkuchen
import scipy


"""legacy code for loading the binary files of labchart. Function is included into GUI"""



#key_list = ['Acoustic mic','EGG','Airflow-Input A']
key_list = ['Ambient mic','EGG','Airflow-Input A']


def build_dict_and_extract_data(f):

    max_samplingrate = 0
    number_records = f.n_records

    #get needed channels
    dict_channels = {'Ambient mic':None,'EGG':None,'Airflow-Input A':None}


    for channel in f.channels:
        if channel.fs[0] > max_samplingrate:
            max_samplingrate = channel.fs[0]


        for key in key_list:
            if key == channel.name:
                dict_channels[key] = channel

    #extract the data and interpolate EGG
    #dict_data = {'Acoustic mic': [], 'EGG': [], 'Airflow-Input A': []}
    dict_data = {'Ambient mic': [], 'EGG': [], 'Airflow-Input A': []}

    for key in  key_list:
        for i in range(number_records):
            if dict_channels[key].fs[i] < max_samplingrate:

                dict_data[key].extend(interpolate_less_sampled_data(dict_channels[key].get_data(i+1),dict_channels[key].fs[i],max_samplingrate))

            else:

                dict_data[key].extend(dict_channels[key].get_data(i + 1))

    print('a')

    segments = aS.silence_removal(np.asarray(dict_data['Ambient mic']), int(max_samplingrate), 0.05, 0.05/2, smooth_window=0.5,weight=0.5,min_duration = 0.5,plot=False)

    df_list = extract_recordings_by_segments(dict_data, segments, max_samplingrate, plot=True)

    return df_list, max_samplingrate


def extract_recordings_by_segments(dict,segments,fs,plot = False):

    dt = 1/fs



    acoustic = dict['Ambient mic']
    egg = dict['EGG']
    airflow = dict['Airflow-Input A']
    additional_sec = 0.45
    modifiy_index = int(additional_sec/dt)
    segments_index = []
    for segment in segments:
        segments_index.append([int(segment[0]/dt)-modifiy_index,int(segment[1]/dt)+modifiy_index])

    #get median dist. between segments
    acoustic_list = []
    egg_list = []
    airflow_list = []
    for segment_indexes in segments_index:
        acoustic_list.append(acoustic[segment_indexes[0]:segment_indexes[1]])
        egg_list.append(egg[segment_indexes[0]:segment_indexes[1]])
        airflow_list.append(airflow[segment_indexes[0]:segment_indexes[1]])

    if plot:
        sampling_rate = fs
        seg_limits = segments

        time_x = np.arange(0, len(acoustic) / float(sampling_rate), 1.0 /
                           sampling_rate)

        plt.subplot(3, 1, 1)
        plt.plot(time_x, acoustic)
        for s_lim in seg_limits:
            plt.axvline(x=s_lim[0], color='red')
            plt.axvline(x=s_lim[1], color='red')
        plt.title('Acoustic')

        plt.subplot(3, 1, 2)
        plt.plot(time_x, egg)
        for s_lim in seg_limits:
            plt.axvline(x=s_lim[0], color='red')
            plt.axvline(x=s_lim[1], color='red')
        plt.title('EGG')

        plt.subplot(3, 1, 3)
        plt.plot(time_x, airflow)
        for s_lim in seg_limits:
            plt.axvline(x=s_lim[0], color='red')
            plt.axvline(x=s_lim[1], color='red')
        plt.title('Airflow')
        plt.show()

    df_list = []
    for a,ai,e in zip(acoustic_list,airflow_list,egg_list):
        time = np.linspace(0, len(a) * dt, len(a),dtype=np.float64)
        df_list.append(
            pd.DataFrame(data = {'acoustic': a, 'airflow': ai, 'egg': e, 'time': time})
        )

    return df_list



def interpolate_less_sampled_data(data,current_fs,desired_fs):

    desired_dt = 1 / desired_fs
    desired_entries = len(data) * int(desired_fs/current_fs)
    time_greatest_resolution= np.linspace(0, desired_entries * desired_dt, desired_entries, endpoint=False)
    current_dt = 1/current_fs

    #TODO change interpolation method?
    time = np.linspace(0,len(data)*current_dt,num=len(data),endpoint=False)


    t, c, k = scipy.interpolate.splrep(time, data, s=0, k=4)
    poly = scipy.interpolate.BSpline(t, c, k)
    data_interpolated = poly(time_greatest_resolution)
    return data_interpolated

def dict_and_save(df_list,fs,subject_name):
    path = r"D:\Uni\Masterarbeit\Scripte\Split_recordings"
    for i,df in enumerate(df_list):
        dict = {'data':df,'info': {'fs':fs}}
        flammkuchen.save(join(path,subject_name+'-'+str(i+1).zfill(2)+'.h5'), dict)

def load_everything(parent_folder):

    """

    :param
    :type
    :param
    :type

    """


    file = 'S007_VOMS_04052017.adicht'
    f = adi.read.read_file(join(parent_folder, file))


    dict,fs = build_dict_and_extract_data(f)

    filenames = file.split('.')

    #dict_and_save(dict,fs,filenames[0])


load_everything(r'D:\Uni\Masterarbeit\Scripte\LabChart_data_for_test')

