from Measurements.Voice_onset.PhonationOnset.Calculation_method import Calculation_method


import numpy as np
import scipy
import matplotlib.pyplot as plt
# import math
import Measurements.Voice_onset.PhonationOnset.util as util
import Measurements.Voice_onset.PhonationOnset.preprocessing as pp
# import flammkuchen as fl
# from numba import njit

#TODO:  Documentation

class VocalRiseTime(Calculation_method):
    """
    Calculates the Vocal Rise Time

    Following:
        Characteristics of Phonatory Function in Singers and Nonsingers With Vocal Fold Nodules
        Stepp et al. 2011
    """
    def __init__(self, window_size=0.08 ,step_size=0.0025 , begin_phonation_onset=0.2, begin_steady_state = 0.8,
                 considered_cycles = 5, F_0_determination_method = 'correlation', centroid_periodogram_db_treshold=-4, segment_window = 0.8, duration_hann_window=0.025,
                 energy_crit=0.05, freq_crit=0.15, prefilter_lowcut=75, prefilter_highcut=1000, prefilter_filter_order=4):
        """
        Constructor
            -initialisation of all hyperparameters

        hyperparameters
            window_size: float, default 0.08
                size (in Seconds) of the RMS window
            step_size: float, default 0.0025
                size (in Seconds) of the steps between RMS calculations
            begin_phonation_onset: float, default 0.2
                percentage at which the phonation is considered to start w.r.t. the max. RMS value
            begin_steady_state: float, default 0.8
                percentage at which the steady state is considered to start w.r.t. the max. RMS value

            -hyperparameter for finding the phonation starting point and limiting the signal to the  segment_window value
            prefilter_lowcut: int, default 75
                lower bound for initial Bandpass filter, used to determine the fundamental frequency from the
                egg Signal
                .. note::
                    according to Orlikof et al. 2012 if the recordings is from a female, the lower bound should
                    be at 125 hz
            prefilter_highcut: int, default 1000
                upper bound for initial Bandpass filter, used to determine the fundamental frequency from the
                egg Signal
                .. note::
                    according to Orlikof et al. 2012 if the recordings is from a male, the upper bound should be
                    at 500 hz
            prefilter_filter_order: int, default 4
                order of the butterworth filter, which performs the initial Bandpass filtering from which the
                fundamental frequency gets determined
                .. warning::
                    if the order gets higher then the default value the filtered result will be NAN
                .. note::
                    according to Orlikof et al. 2012 the signal should be filtered with an
                    256-order Roark-Escabí window
                        B-spline design of maximally flat and prolate spheroidal-type FIR filters,
                        Roark et al. 1999
                    as replacement a butterworth filter is implemented
             F_0_determination_method: string, default 'correlation'
                defines how the fundamental frequency get calculated -> 'fft_peak', 'correlation',
                'centroid_periodogram'
            centroid_periodogram_db_treshold: int default -4,
                treshold at which peaks in the periogram are considered for centroid calculation
                ..note::
                Has to be negative or 0!
                Just maters using the centroid_periodogram (choosen with F_0_determination_method) when determining f_0
                If there is any trouble with finding a stable fundamental frequency, it is worth to try some other
                values for this treshhold
            duration_hann_window: float, default 0.025
                the size of the hann window in seconds, used in the preprocessing step for defining the starting point
                of the phonation
            energy_crit: float, default 0.05
                percentage value that the oscilation peaks have to reach at least, to be considered as a starting point
                of the phonation
            freq_crit: float, default 0.15
                used for the frequency stability criteria
                percentage value that the difference between to peaks have to be around the median frequency to be
                considered as a starting point
            segment_window, float default 0.2
                restricts the signal from the determined starting point of the phonation up to the value of
                segment_window in seconds


            considered_cycles: int, default 5
                defines how many periods are considered for calculation of the relativ onset time


        Attributes:
        -Intermediate results-

            framerate: int
                self explanatory
            begin_onset: int
                index where the RMS value reaches the percentage value of "begin_phonation_onset"
            steady_state: int
                index where the RMS value reaches the percentage value of "begin_steady_state"
            correspondance_begin_onset: int
                index at which point does the phonation starts in the non-subsampled signal
            correspondance_steady_state: int
                index at which point does the steady state starts in the non-subsampled signal
            num_samples_cycle: int
                number of average sampled between two cycle peaks
            step_adjusted: ndarray
                downsampled time array after the window root mean square calculation
            window_rms: ndarray
                result of the the window root mean square calculation
            F_0: float
                fundamental frequency
            acoustic: ndarray
                the normalized acoustic signal
            time_array: ndarray
                the duration array of the acoustic signal


            -result attributes-

            norm_time: float
                normalized/relativ onset time, wrt. to the duration of the input parameter consideres_cycles
            onset_label: ndarray
                one hot encoded array which denotes the onset process, also includes the duration of
                the input parameter consideres_cycles
           time: float
                voice onset time

        """

        self.window_placements = ['left', 'middle', 'right']


        self.prefilter_lowcut = prefilter_lowcut
        self.prefilter_highcut = prefilter_highcut
        self.prefilter_filter_order = prefilter_filter_order
        self.window_size = window_size  # seconds
        self.step_size = step_size
        self.segment_window = segment_window
        self.duration_hann_window = duration_hann_window
        self.energy_crit = energy_crit
        self.freq_crit = freq_crit
        self.begin_phonation_onset = begin_phonation_onset
        self.begin_steady_state = begin_steady_state
        self.F_0_determination_method = F_0_determination_method
        self.centroid_periodogram_db_treshold = centroid_periodogram_db_treshold
        self.considered_cycles = considered_cycles

        #attributes

        self.acoustic = None
        self.time_array = None
        self.begin_onset = None
        self.steady_state = None
        self.correspondance_begin_onset = None
        self.correspondance_steady_state = None
        self.num_samples_cycle = None
        self.step_adjusted = None
        self.window_rms = None
        self.framerate = None
        self.time = None
        self.norm_time = None
        self.onset_label = None
        self.F_0 = None

    def compute(self, acoustic, framerate, window_shift='right'):

        """
       Calculates the Vocal Rise Time


       :param time: time of the acoustic signal
       :type time: ndarray
       :param acoustic: acoustic signal
       :type acoustic: ndarray
       :param framerate: sampling frequency
       :type time: int
       :param window_shift: where the calculation center of the RMS is placed -> Options: ['left','middle','right']
       :type method: string
       :param plot: switch if the calculation process should be plotted
       :type plot: boolean
       """
        assert window_shift in self.window_placements, "Wrong method! Use one of the following methods: " + ' '.join(
            self.window_placements) + ' as String'
        #Some silence (500ms to the left and to the right) must be added because that´s how the stupid function works.
        #Apparenly the signal must be like 600ms long in order to compute this right
        s1 = np.zeros(len(acoustic)+framerate)
        s1[int(framerate/2):int(len(s1)-(framerate/2))] = acoustic
        acoustic = s1.copy()
        self.framerate = framerate
        self.acoustic = acoustic
        time = np.arange(0,len(acoustic)/framerate,1/framerate)
        self.time_array = time

        # from scipy.fft import fft, fftfreq
        # from_start = 16000
        # from_end = 14000
        # plot_time = self.time_array[from_start:-from_end]-self.time_array[from_start]
        # plot_acc = self.acoustic[from_start:-from_end]
        # plot_acc = pp.normalize(plot_acc,'z-scoring')
        # plt.subplot(211)
        # plt.title('Time Domain')
        # plt.plot(plot_time,plot_acc)
        # plt.xlabel('Duration [s]')
        # plt.ylabel('Norm. Amplitude')
        # plt.subplot(212)
        # plt.title('Frequency Domain')
        # yf = fft(plot_acc)
        # #xf = fftfreq(len(plot_acc), plot_time[-1])[:len(plot_acc) // 2]
        # plt.plot(2.0 /len(plot_acc)  * np.abs(yf[0:len(plot_acc) // 2]))
        # plt.xticks(np.arange(0,len(plot_acc),1)[0::200])
        # plt.xlabel('Frequency [1/s]')
        # plt.ylabel('Fourier Coefficient')
        # plt.grid()
        # plt.show()
        #self.acoustic = pp.normalize(acoustic,'z-scoring')

        #converting the windowduration into how many sample are considered
        window = int(np.round((self.window_size) / (1 / framerate)))
        stepsize = int(np.round((self.step_size) / (1 / framerate)))
        segmentsize = int(np.round((self.segment_window) / (1 / framerate)))


        nyq = 0.5 * framerate

        b, a = scipy.signal.butter(self.prefilter_filter_order,
                                   [self.prefilter_lowcut / nyq, self.prefilter_highcut / nyq],
                                   btype='band', analog=False)

        prefiltered_acoustic = scipy.signal.filtfilt(b, a, self.acoustic)

        self.r_0 = pp.Onset_detection_binary_crit(self.time_array, prefiltered_acoustic, framerate,
                                                  duration_hann_window=self.freq_crit, energy_crit=self.energy_crit,
                                                  freq_crit=self.freq_crit)

        self.step_adjusted, self.window_rms = self.moving_average(time[:self.r_0+segmentsize], self.acoustic[:self.r_0+segmentsize], window, stepsize, window_shift)
        #self.step_adjusted, self.window_rms = self.moving_average(time, acoustic, window, stepsize, window_shift)
        #[:int(np.ceil(self.r_0 / stepsize + segmentsize / stepsize))]
        #where does the onset begin and when does it end
        self.begin_onset = next(x for x, val in enumerate(self.window_rms)
                           if val >= self.begin_phonation_onset * np.max(self.window_rms))
        self.steady_state = next(x for x, val in enumerate(self.window_rms)
                            if val >= self.begin_steady_state * np.max(self.window_rms))


        #RMS got calculated at a downsampled signal, where is the onset/steady state in the original time array
        self.correspondance_begin_onset = np.where(time >= self.step_adjusted[self.begin_onset])[0][0]
        self.correspondance_steady_state = np.where(time >= self.step_adjusted[self.steady_state])[0][0]
        #for determination of the cycle length
        self.F_0 = pp.get_fundamental_frequency(pp.normalize(acoustic,'c_scoring'), framerate, mode=self.F_0_determination_method,
                                           centroid_periodogram_db_treshold= self.centroid_periodogram_db_treshold, plot=False)
        self.num_samples_cycle = int(1 / (self.F_0  * (1/framerate)))

        self.time = self.step_adjusted[self.steady_state] - self.step_adjusted[self.begin_onset]

        self.onset_label = util.get_onset_label(acoustic, self.correspondance_begin_onset, self.correspondance_steady_state
                                                , self.num_samples_cycle, self.considered_cycles)
        
        
        #Remove the silence added at the beginning of this function.
        self.onset_label = self.onset_label[int(framerate/2):int(len(self.onset_label)-(framerate/2))]

        self.norm_time = util.get_relativ_time(self.correspondance_begin_onset, self.correspondance_steady_state
                                                , self.num_samples_cycle, self.considered_cycles)

    #
    def moving_average(self,time, x, window, steps, shift='right'):

        """
        calculates a windowed root mean square
        Args:
            time: ndarry
                time array
            x: ndarry
                signal from which the rms should be calculated
            window: int
                size of the moving average window
            steps: int
                step size of the moving average window
            shift: String, default: 'right'
                shits the output of the rms calculation to the first/middle/last element of the output array

        Returns: ndarray

        """
        time_adjusted = []
        rms = []

        for i in range(0, len(x), steps):
            if shift == 'middle':
                lb = window // 2 if i >= window // 2 else 0
                hb = window // 2 if i < x.shape[0] - window // 2 else x.shape[0]
                rms.append(np.sqrt(np.mean(x[i-lb:i+hb+1] ** 2)))
                time_adjusted.append(time[i])
            elif shift == 'left':
                hb = window  if i < x.shape[0] - window  else x.shape[0]
                rms.append(np.sqrt(np.mean(x[i:i+hb+1] ** 2)))
                time_adjusted.append(time[i])
            else:
                lb = window if i >= window else 0
                rms.append(np.sqrt(np.mean(x[i - lb:i + 1] ** 2)))
                time_adjusted.append(time[i])

        return np.asarray(time_adjusted), np.asarray(rms)

    def set_new_hyperparameter(self,dict):
        '''changes the hyperparameters'''

        self.window_size=dict['window_size']
        self.step_size=dict['step_size']
        self.begin_phonation_onset=dict['begin_phonation_onset']
        self.begin_steady_state=dict['begin_steady_state']
        self.considered_cycles=dict['considered_cycles']
        self.F_0_determination_method=dict['F_0_determination_method']
        self.centroid_periodogram_db_treshold=dict['centroid_periodogram_db_treshold']
        self.duration_hann_window=dict['duration_hann_window']
        self.energy_crit=dict['energy_crit']
        self.freq_crit=dict['freq_crit']
        self.segment_window=dict['segment_window']
        self.prefilter_lowcut=dict['prefilter_lowcut']
        self.prefilter_highcut=dict['prefilter_highcut']
        self.prefilter_filter_order=dict['prefilter_filter_order']

# #
# data = fl.load(r"..\Split_recordings\S001_VOMS_280417-03.h5")
# acoustic = data['data']['acoustic'].values
# framerate = data['info']['fs']
# time = data['data']['time'].values
# vrt = VocalRiseTime()
# vrt.compute(time,acoustic,framerate,plot=True)