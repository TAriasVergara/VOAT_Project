from Measurements.Voice_onset.PhonationOnset.Calculation_method import Calculation_method


import numpy as np
import scipy
import matplotlib.pyplot as plt
# import math
import Measurements.Voice_onset.PhonationOnset.util as util
import Measurements.Voice_onset.PhonationOnset.preprocessing as pp
# import flammkuchen as fl
# import os

#TODO: check in periodogram method the considering threshold( currently -4 db, paper -7 db, could be replaced by highest value)

class VocalAttackTime(Calculation_method):
    """
    Calculates the Vocal attack time by fitting measuring the timelag of the envelope function between the accoustic
    and the egg signal:

    Following:
        -Validation of a Glottographic Measure of Vocal Attack, Orlikof et al. 2009

        -Measures of Vocal Attack Time for Healthy Young Adults, Orlikof et al. 2012
            -the algorithm is mostly recreated from this paper, but some methods are not well described or time
            consuming to implement, therefore workaround implementations are used
    """
    def __init__(self,
                 prefilter_lowcut=75,
                 prefilter_highcut=1000,
                 prefilter_filter_order=4,
                 duration_hann_window=0.025,
                 energy_crit=0.15,
                 freq_crit=0.15,
                 extraction_window =0.6,
                 F_0_determination_method='centroid_periodogram',
                 centroid_periodogram_db_treshold=-4,
                 filter_option='firwin_bandpass',
                 bandpass_filter_range=0.4,
                 bandpass_filter_order=1500,
                 delta_derivation=0.01,
                 cut_away_range=0.45,
                 considered_cycles = 5):

        """
        Constructor
            -initialisation of all hyperparameters

            -seeting.py provides all hyperparameters as dict

        Parameters
            -initial filtering for fundamental frequency determination from the EGG signal-

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

            -definition for the phonation starting point-

            duration_hann_window: float, default 0.025
                the size of the hann window in seconds, used in the preprocessing step for defining the starting point
                of the phonation
            energy_crit: float, default 0.15
                percentage value that the oscilation peaks have to reach at least, to be considered as a starting point of
                the phonation
            freq_crit: float, default 0.15
                used for the frequency stability criteria
                percentage value that the difference between to peaks have to be around the median frequency to be
                considered as a starting point

             -fundamental frequency determination-

             F_0_determination_method: string, default 'centroid_periodogram'
                defines how the fundamental frequency get calculated -> 'fft_peak', 'correlation', 'centroid_periodogram'
                ..note::
                    centroid_periodogram is by far the most stable method
                    according to roark et al. 2012 the range from which the frequency gets determined is - 7 dB
                    implemented is -4 dB
            centroid_periodogram_db_treshold: int default -4,
                treshold at which peaks in the periogram are considered for centroid calculation
                ..note::
                Has to be negative or 0!
                Just maters using the centroid_periodogram (choosen with F_0_determination_method) when determining f_0
                If there is any trouble with finding a stable fundamental frequency, it is worth to try some other
                values for this treshhold

            filter_option: string, default 'firwin_bandpass'
                final filtering to obtain a stable envelope function from the hilbert transform
                Options -> ['recursive_detrending', 'firwin_bandpass']
                ..note::
                    #TODO: validate recursive detrending
                    Following Roark et al. 2012 the method should be 'recursive_detrending', but the
                    implementation has to be validated
            bandpass_filter_range: float, default 0.4
                    percentage value for the lower & upper bandpass ranges w.r.t. the fundamental frequency
                    For example:
                    cutoff frequency: F_0 *(1.0 +/- 0.1)
            bandpass_filter_order: int, default 1500
                    order of the firwin bandpass filter
                    ..note::
                        this parameter only is used when the parameter filter_option = butter_bandpass is used
                -VAT determination parameters-
            delta_derivation float, default 0.01
                stepssize (in seconds) used in the numerical derivation of the signal, like:
                    derivation = (x(t) - x(t+h))/h with h = 0.01
            cut_away_range=0.45,
                    percentage of cut off from the right and left side of the derivation


            considered_cycles: int, default 5
                defines how many periods are considered for calculation of the relativ onset time
                also includes n periods as pre start onset phase and post steady state phase to the cut out signal

        Attributes
            -Intermediate results-

            prefiltered_egg: np.ndarray
                filtered EGG signal for phonation start point and fundamental frequency determination
            start_point_phonation_index: int
                index at which the phonation starts
            f_0: float
                fundamental frequency
            num_samples_cycle: int
                length of points between to peaks
            fitered_acoustic: np.ndarray
                acoustic signal filtered around f_0
            filtered_egg: np.ndarray
                EGG signal filtered around f_0
            extracted_acoustic: np.ndarray
                cut out acoustic signal
            extracted_time: np.ndarray
                cut out time array
            extracted_egg: np.ndarray
                cut out EGG signal
            hilbert_acoustic: np.ndarray
                hilbert transform of the cut out and filtered acoustic signal
            self.hilbert_egg: np.ndarray
                hilbert transform of the cut out and filtered EGG signal
            egg_deriv: np.ndarray
                derivation of the hilbert transformed EGG signal
            acoustic_deriv: np.ndarray
                derivation of the hilbert transformed acoustic signal
            cross_corr: np.ndarray
                result of the cross correlation
            peak: int
                index of the maximum of the cross correlation
            corr_time: np.ndarray
                time array for the cross correlation
            r_0: int
                index for the onset label determination, where does the phonation starts in the acoustic signal
            r_sat:
                index for the onset label determination, where does the starty state starts in the acoustic signal


            -return attributes-

            norm_time: float
                normalized/relativ onset time, wrt. to the duration of the input parameter considered_cycles
            onset_label: ndarray
                one hot encoded array which denotes the onset process, also includes the duration of
                the input parameter considered_cycles
           time: float
                voice onset time

        """
        self.bandpass_filter_mode = ['recursive_detrending','firwin_bandpass']

        self.filter_option = filter_option

        self.prefilter_lowcut = prefilter_lowcut
        self.prefilter_highcut = prefilter_highcut
        self.prefilter_filter_order = prefilter_filter_order
        self.duration_hann_window = duration_hann_window
        self.energy_crit = energy_crit
        self.freq_crit = freq_crit
        self.extraction_window = extraction_window

        self.F_0_determination_method = F_0_determination_method
        self.centroid_periodogram_db_treshold = centroid_periodogram_db_treshold
        self.bandpass_filter_range = bandpass_filter_range
        self.filter_option = filter_option
        self.bandpass_filter_order = bandpass_filter_order
        self.delta_derivation = delta_derivation
        self.cut_away_range = cut_away_range

        self.considered_cycles = considered_cycles
        
        #calculation variables:

        self.acoustic = None
        self.egg = None
        self.time_array = None
        self.len_extraction_window = None


        self.prefiltered_egg = None
        self.start_point_phonation_index = None
        self.f_0 = None
        self.num_samples_cycle = None
        self.fitered_acoustic = None
        self.filtered_egg = None
        self.extracted_acoustic = None
        self.extracted_time = None
        self.extracted_egg = None
        self.egg_deriv = None
        self.acoustic_deriv = None
        self.hilbert_acoustic = None
        self.hilbert_egg = None
        self.cross_corr = None
        self.peak = None
        self.corr_time = None
        self.r_0 = None
        self.r_sat = None

        #return variables
        self.norm_time = None
        self.onset_label = None
        self.time = None

    def compute(self,acoustic,egg,framerate,plot=False):

        """
        performs the the calculation of the Vocal Attack Time

        :param acoustic: acoustic signal
        :type acoustic: ndarray
        :param egg: electroglottographic signal
        :type egg: ndarray
        :param plot: switch if the calculation process should be plotted
        :type plot: boolean
        :param framerate: sampling frequency
        :type time: int
        :param plot: switch if the calculation process should be plotted
        :type plot: boolean
        """
        #normalize data
        acoustic = pp.normalize(acoustic)
        egg = pp.normalize(egg)
        
    
        #Some silence (500ms to the left and to the right) must be added because that´s how the stupid function works.
        #Apparenly the signal must be like 600ms long in order to compute this right
        s1 = np.zeros(len(acoustic)+framerate)
        s2 = np.zeros(len(egg)+framerate)
        s1[int(framerate/2):int(len(s1)-(framerate/2))] = acoustic
        acoustic = s1.copy()
        s2[int(framerate/2):int(len(s2)-(framerate/2))] = egg
        egg = s2.copy()
        
        self.acoustic = acoustic
        self.egg = egg
        self.time_array = np.arange(0,len(acoustic)/framerate,1/framerate)
        #bandpass prefiltering to extract the starting point out of the egg signal
        #________________________________________
        #TODO: Placeholder filtering:
        #Original method had a specific method, see: Measures of Vocal Attack Time for Healthy Young Adults, Roark et al.
        #They used a ideal filter window, defined some criteria points on that rectangular fct and interpolated those with
        #a spline fct to design that filter
        #also they adjusted the bandpass filterranges by gender

        nyq = 0.5 * framerate

        b, a = scipy.signal.butter(self.prefilter_filter_order, [self.prefilter_lowcut / nyq, self.prefilter_highcut / nyq],
                                   btype='band', analog=False)

        self.prefiltered_egg = scipy.signal.filtfilt(b, a, egg)
        #________________________________________


        #where does the phonation starts
        self.start_point_phonation_index = pp.Onset_detection_binary_crit(self.time_array, self.prefiltered_egg, framerate,
                                    duration_hann_window=self.duration_hann_window,
                                    energy_crit=self.energy_crit, freq_crit=self.freq_crit)

        #calculate the distance between samples
        dt_signal = 1 / framerate

        #how many samples are needed for the extraction
        #TODO: catch exception if the signal could not be extracted to fit the 300ms left from the starting point in it
        self.len_extraction_window = int(np.round(self.extraction_window / dt_signal))

        #extract the data out of the signal (default 0.6 ms window)
        self.extracted_egg = self.__cut_signal(egg, self.start_point_phonation_index, self.len_extraction_window)
        self.extracted_acoustic = self.__cut_signal(acoustic, self.start_point_phonation_index, self.len_extraction_window)
        self.extracted_time = self.__cut_signal(self.time_array, self.start_point_phonation_index, self.len_extraction_window)
        extracted_prefiltered_egg = self.__cut_signal(self.prefiltered_egg, self.start_point_phonation_index, self.len_extraction_window)

        #TODO: do not know if this is still necessary
        extracted_prefiltered_egg = pp.normalize(extracted_prefiltered_egg)

        #calculate the fundamental frequency out of the egg signal right of the starting point
        self.f_0 = pp.get_fundamental_frequency(extracted_prefiltered_egg[int(self.len_extraction_window/2):],framerate,
                    centroid_periodogram_db_treshold=self.centroid_periodogram_db_treshold,
                                                mode=self.F_0_determination_method, plot=False)

        assert self.filter_option in self.bandpass_filter_mode, "Wrong method! Use one of the following methods: " + ' '.join(self.bandpass_filter_mode) + ' as String'
        if self.filter_option == 'recursive_detrending':
            self.filtered_acoustic = pp.recursive_detrending_bandpass_filter(self.extracted_time, self.extracted_acoustic, self.bandpass_filter_range, framerate, self.f_0, plot=False)
            self.filtered_egg = pp.recursive_detrending_bandpass_filter(self.extracted_time, self.extracted_egg, self.bandpass_filter_range, framerate, self.f_0, plot=False)
        else:
        #bandpass filter the signal
            self.filtered_acoustic = pp.filter_signal_firwin_bandpass(self.extracted_acoustic, self.f_0 * (1.0 - self.bandpass_filter_range),
                                self.f_0 * (1.0 + self.bandpass_filter_range), self.bandpass_filter_order, framerate=framerate)
            self.filtered_egg = pp.filter_signal_firwin_bandpass(self.extracted_egg, self.f_0 * (1.0 - self.bandpass_filter_range),

                                self.f_0 * (1.0 + self.bandpass_filter_range), self.bandpass_filter_order, framerate=framerate)

        #get envelope function
        self.hilbert_acoustic = pp.get_hilbert_transform(self.filtered_acoustic)
        self.hilbert_egg = pp.get_hilbert_transform(self.filtered_egg)
        #
        # hilbert = scipy.signal.hilbert(self.filtered_acoustic, N=None, axis=- 1)
        # plt.plot(self.extracted_time, self.filtered_acoustic, color='blue',label = 'filtered acoustic signal')
        # plt.plot(self.extracted_time, hilbert.imag, color='orange', label = 'hilbert transform of the filtered acoustic signal')
        # plt.plot(self.extracted_time, self.hilbert_acoustic, color='black', label='hilbert envelope')
        # plt.title('hilbert envelope of a signal')
        # plt.xlabel('Duration [s]')
        # plt.ylabel('Normalized amplitude [%]')
        # plt.legend()
        # plt.show()

        #first orderderivate with additional filtering( -> dt = 10ms)
        delta_derivation = self.delta_derivation
        delta_t = int(np.round(delta_derivation / dt_signal))
        derivation_array = np.zeros(delta_t)
        derivation_array[0] = 1
        derivation_array[-1] = -1

        self.acoustic_deriv = np.convolve(self.hilbert_acoustic, derivation_array, mode='same')
        acoustic_deriv_cut = self.acoustic_deriv[int(self.cut_away_range * len(self.filtered_acoustic)):
                                            -int(self.cut_away_range * len(self.filtered_acoustic))]
        #perform the crosscorrelation
        self.egg_deriv = np.convolve(self.hilbert_egg, derivation_array, mode='same')
        #cut the correlation result from left and right -> filtering and the hilberttransform resulting in slope at the
        #beginning and end of the signal, these slope are independend from the time lag at the middle
        #if the slopes are left in the signal they will adulterate the correlation result
        egg_deriv_cut = self.egg_deriv[int(self.cut_away_range * len(self.filtered_acoustic)):
                                  -int(self.cut_away_range * len(self.filtered_acoustic))]

        #determination of the VAT
        cut_step = self.extracted_time[
                   int(self.cut_away_range* len(self.filtered_acoustic)):-int(self.cut_away_range * len(self.filtered_acoustic))]
        self.cross_corr = scipy.signal.correlate(acoustic_deriv_cut, egg_deriv_cut, mode='full', method='direct')

        self.peak = np.where(self.cross_corr == np.max(self.cross_corr))[0][0]
        self.corr_time = np.linspace(-(cut_step[-1] - cut_step[0]), (cut_step[-1] - cut_step[0]), len(self.cross_corr))

        self.time = self.corr_time[self.peak]

        self.filtered_acoustic_r_0 = pp.filter_signal_firwin_bandpass(self.acoustic,
                                                                  self.f_0 * (1.0 - self.bandpass_filter_range),
                                                                  self.f_0 * (1.0 + self.bandpass_filter_range),
                                                                  self.bandpass_filter_order, framerate=framerate)

        self.r_0 = pp.Onset_detection_binary_crit(self.time_array, self.filtered_acoustic_r_0, framerate,
                                    duration_hann_window=self.duration_hann_window,
                                    energy_crit=self.energy_crit, freq_crit=self.freq_crit)


        self.num_samples_cycle = int(1 / (self.f_0 * dt_signal))

        # self.onset_label = util.get_onset_label(egg, self.start_point_phonation_index, self.start_point_phonation_index +
        #                                         int(self.time / dt_signal), self.num_samples_cycle, self.considered_cycles)

        self.onset_label = util.get_onset_label(self.filtered_acoustic_r_0, self.r_0, int(self.r_0+self.time/dt_signal),
                                                         self.num_samples_cycle, self.considered_cycles)
        
        #Remove the silence added at the beginning of this function.
        self.onset_label = self.onset_label[int(framerate/2):int(len(self.onset_label)-(framerate/2))]
        
        self.norm_time = util.get_relativ_time(self.start_point_phonation_index, self.start_point_phonation_index +
                                                int(np.abs(self.time) / dt_signal), self.num_samples_cycle, self.considered_cycles)


    def __cut_signal(self,signal,midpoint,range):

        """
        Helperfunction, takes the signal and cuts the signal around the midpoint with half of the range to the left
        and to the right

        """
        return signal[int(midpoint - range / 2):int(midpoint + range / 2)]


    def set_new_hyperparameter(self, dict):

        '''changes the hyperparameters'''

        self.prefilter_lowcut = dict['prefilter_lowcut']
        self.prefilter_highcut = dict['prefilter_highcut']
        self.prefilter_filter_order = dict['prefilter_filter_order']
        self.duration_hann_window = dict['duration_hann_window']
        self.energy_crit = dict['energy_crit']
        self.freq_crit = dict['freq_crit']
        self.extraction_window = dict['extraction_window']
        self.F_0_determination_method = dict['F_0_determination_method']
        self.centroid_periodogram_db_treshold = dict['centroid_periodogram_db_treshold']
        self.filter_option = dict['filter_option']
        self.bandpass_filter_range = dict['bandpass_filter_range']
        self.bandpass_filter_order = dict['bandpass_filter_order']
        self.delta_derivation = dict['delta_derivation']
        self.cut_away_range = dict['cut_away_range']
        self.considered_cycles = dict['considered_cycles']
