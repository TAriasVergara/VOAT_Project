from Measurements.Voice_onset.PhonationOnset.Calculation_method import Calculation_method

import numpy as np
import scipy
import matplotlib.pyplot as plt
import math

import Measurements.Voice_onset.PhonationOnset.util as util
import Measurements.Voice_onset.PhonationOnset.preprocessing as pp
# import lmfit
# from numba import njit


class VoiceOnsetTime(Calculation_method):
    """
    Calculates the Voice Onset Time by fitting an envelope function to the oscillation peaks:

    Either with:

    The Kunduk Method: -> Fitting of a polynom of spline function

        -Effects of Volume, Pitch, and Phonation Type on Oscillation Initiation and Termination Phases
        Investigated With High-speed Videoendoscopy, Kunduk et al. 2017

        -Evaluation of Analytical Modeling Functions for the Voice Onset Process, Peterman et. al. 2016

    The Mergel Method: -> Fitting of an exponential saturation function

        -Phonation onset: Vocal fold modeling and high-speed glottography, Mergel et al. 1998

    """

    def __init__(self,
                 poly_degree=4,
                 steady_state_requirement=0.90,
                 begin_onset=0.1,
                 kunduk_calculation_method='bspline',
                 first_significant_peak_crit=0.5,
                 first_significant_peak_option='on',

                 begin_onset_mergel=0.322,
                 steady_state_mergel=0.678,
                 mergel_calculation_method='fit: a, r_0, r_sat',

                 reduce_signal=0.3,
                 F_0_determination_method='correlation',
                 centroid_periodogram_db_treshold=-4,
                 F_0_bandpass_filter_range=0.10,
                 filter_order=3,
                 duration_hann_window=0.025,
                 energy_crit=0.05,
                 freq_crit=0.15,
                 r_sat_consideration_crit=0.95,
                 cma_peaks_window=5,
                 considered_cycles=5
                 ):

        """
        Constructor
            -initialisation of all hyperparameters

        Parameters
            -Hyperparameters for the Kunduk calculation method-

            reduce_signal: float, default 0.3
                cuts the chosen value in seconds from both ends of the signal
            poly_degree: int, default 4
                order of the polynominial fitting function
            steady_state_requirement: float, default 0.90
                percentage wrt. to the maximum amplitude value at which the steady state function is reached
            begin_onset: float, default 0.1
                percentage wrt. to the maximum amplitude value at which the phonation process starts
            kunduk_calculation_method: string, default 'poly'
                determines which sort of polynominial function gets to be fitted -> ['poly', 'bspline', 'cspline']
            first_significant_peak_crit, float default 0.6
                if the polynom ocsillates and has a maximum peak point
                percentage value at which a polynom peaks gets considered to be significant to reduce the onset time
                calculation from the begin of the onset process to this peak
            -Hyperparameters for the Mergel calculation method-

            begin_onset_mergel: float, default 0.322
                percentage wrt. to the maximum amplitude value at which the phonation process starts
                this value is normaly fixed by definition, should only be changed with caution!
            steady_state_mergel: float, default 0.678
                percentage wrt. to the maximum amplitude value at which the steady state function is reached
                this value is normaly fixed by definition, should only be changed with caution!
            mergel_calculation_method: string, default 'three_free_parameter'
                determines how many free parameters to be fitted -> ['three_free_parameter', 'one_free_parameter']

            -Generell hyperparameters which are used in either of the calculation methods-
            reduce_signal: float, default 0.3
                cuts the signal from the left and right sight according to reduce signal in seconds and leaves only the
                signal in between.
                ..note::
                This hyperparamter was used to get rid of leftover signals during the supposedly silent parts in
                between events, since the events got cut out with 0.45 seconds overhang from their recordings
                ..warning::
                If you are using this parameter calculation class outside the developed GUI, it is highly recommended
                to set this value to 0.0.
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
            filter_order: int, default 3
                order of the butterworth bandpass filter.
                ..warning::
                    if the order gets to high the filtered signal will ne NAN
            F_0_bandpass_filter_range: float, default 0.1
                percentage at which the signal gets filtered around the fundamental frequency
                example for 0.1  cutoffrequency = F_0 * (1+/- 0.1)
            cma_peaks_window: int, default 5
                how many oscilation peaks will be included into the moving average
                the moving average will be used to reduce the influence of outliers when defining the r_sat value
            r_sat_consideration_crit: float, default 0.95
                The first oscillation peak that reaches this percent value w.r.t. the maximum signal amplitude
                gets choosen as r_sat
                The peaks from r_0 to r_sat + half of the cma_peaks_window size are used to fit
                sthe envelope functions to



            considered_cycles: int, default 5
                defines how many periods are considered for calculation of the relativ onset time
                also includes n periods as pre start onset phase and post staeady state phase to the cut out signal

                -definition for the phonation starting point-

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

        Attributes:

            -Intermediate results-
            F_0: float
                fundamental frequency, determined with the hyperparameter "F_0_determination_method"
            r_0: int
                at which index does the phonation start
            peaks: ndarray
                all oscillation peaks that have the same or a greater amplitude than the value at r_0
            r_sat: int
                at which index w.r.t. to peaks has the phonation reached an amplitude that can be considered as saturated
            begin_onset_index: int
                at which index does the algorithm consider the phonation to start
            begin_steady_state_index: int
                at which index does the algorithm consider the phonation to reach its steady state
            mean_dist_peaks: int
                the mean distance between the oscillation peaks
            filtered_acoustic: ndarray
                the bandpass filtered acoustic signal
            acoustic: ndarray
                the normalized acoustic signal
            evaluated_fct: ndarray
                the resulting fitting function
            first_significant_peak: int
                only of interest if 'first_significant_peak_option' is 'on', replaces the r_sat value functionwise
                first ocouring peak of the attribute peak, which is above the 'first_significant_peak_crit' value
            time_array: ndarray
                the duration array of the acoustic signal
            filtered_signal_evaluation_points: ndarray
                the acoustic signal values of the peaks, which are used to fit the kunduk envelope function
            steps_evaluation_points: ndarray
                the corresponding timesteps of "filtered_signal_evaluation_points" used to fit the envelope function
            steady_state_poly: int
                index of the stready state begin in the kunduk envelope function
            begin_onset_poly: int
                index of the onset begin in the kunduk envelope function
            steps_evaluation: ndarray
                every time point in between the first and last peak of filtered_signal_evaluation_points
                used to obtain the polynom after fitting
            evaluated_poly: ndarray
                every acoustic signal point in between the first and last peak of filtered_signal_evaluation_points
                used to obtain the polynom after fitting


            -result attributes-

            norm_time: float
                normalized/relativ onset time, wrt. to the duration of the input parameter consideres_cycles
            onset_label: ndarray
                one hot encoded array which denotes the onset process, also includes the duration of
                the input parameter consideres_cycles
            time: float
                voice onset time

        """

        self.calculation_methods = ['mergel', 'kunduk']
        self.kunduk_calculation_methods = ['bspline', 'cspline', 'poly']
        self.mergel_calculation_methods = ['fit: a', 'fit: a, r_0, r_sat']
        self.toggle_options = ["on", "off"]

        # standard parameters for the kunduk_method
        self.poly_degree = poly_degree
        self.begin_onset = begin_onset
        self.steady_state_requirement = steady_state_requirement
        self.kunduk_calculation_method = kunduk_calculation_method
        self.first_significant_peak_option = first_significant_peak_option
        self.first_significant_peak_crit = first_significant_peak_crit

        # standard parameters for the mergel_method
        self.begin_onset_mergel = begin_onset_mergel
        self.steady_state_mergel = steady_state_mergel
        self.mergel_calculation_method = mergel_calculation_method

        # general parameters used by both calculation Methods
        self.reduce_signal = reduce_signal
        self.F_0_determination_method = F_0_determination_method
        self.centroid_periodogram_db_treshold = centroid_periodogram_db_treshold

        self.F_0_bandpass_filter_range = F_0_bandpass_filter_range
        self.filter_order = filter_order
        self.duration_hann_window = duration_hann_window
        self.energy_crit = energy_crit
        self.freq_crit = freq_crit
        self.r_sat_consideration_crit = r_sat_consideration_crit

        # TODO: should contain a dict of simple float with the length -> right now not in use
        # self.restrict_length = restrict_length

        # generall parameters
        self.cma_peaks_window = cma_peaks_window
        self.considered_cycles = considered_cycles

        # hopf burification parameters get defined during runtime -> mergel Method

        # calculation variables:
        self.F_0 = None
        self.r_0 = None
        self.peaks = None
        self.r_sat = None
        self.fitting_peaks = None
        self.begin_onset_index = None
        self.begin_steady_state_index = None
        self.mean_dist_peaks = None
        self.filtered_acoustic = None
        self.acoustic = None
        self.evaluated_fct = None
        self.first_significant_peak = None
        self.time_array = None
        self.steps_evaluation_points = None
        self.filtered_signal_evaluation_points = None
        self.steady_state_poly = None
        self.begin_onset_poly = None
        self.steps_evaluation = None
        self.evaluated_poly = None

        # TODO: get the attributes right
        # return variables
        self.norm_time = None
        self.onset_label = None
        self.time = None

    def compute(self, acoustic, framerate, method='kunduk'):

        """
        performs the preprocessing and calculates the voice onset time with either the kunduk or mergel method

        -normalizes the acoustic signal between -1 and 1
        -reduces the acoustic signal from the left with the duration reduce_signal to remove the influence of artefacts
        -calculates the fundamental frequency
        -bandpass filters the signal around the fundamental frequency
        -detects the starting point of the phonation r_0
        -detects the saturation point r_sat of the phonation


        :param time: time of the acoustic signal
        :type time: ndarray
        :param acoustic: acoustic signal
        :type time: ndarray
        :param framerate: sampling frequency
        :type time: int
        :param method: which calculation method to be performed -> Options: ['kunduk','mergel']
        :type method: string
        :param plot: switch if the onset time calculation should be plotted
        :type plot: boolean
        """
        #Some silence (500ms to the left and to the right) must be added because that´s how the stupid function works.
        #Apparenly the signal must be like 600ms long in order to compute this right
        s1 = np.zeros(len(acoustic)+framerate)
        s1[int(framerate/2):int(len(s1)-(framerate/2))] = acoustic
        acoustic = s1.copy()
        
        self.acoustic = pp.normalize(acoustic)
        time = np.arange(0,len(acoustic)/framerate,1/framerate)
        self.time_array = time
        
        # # self.time_array = time
        # if self.reduce_signal != 0.0:
        #     points_to_reduce = int(framerate * self.reduce_signal)

        #     # reduce the signals length for display purposes
        #     self.acoustic = self.acoustic[points_to_reduce:]
        #     self.time_array = time[points_to_reduce:]

        # throws assert if calculation mode is invalid
        assert method in self.calculation_methods, "Wrong method! Use one of the following methods: " + ' '.join(
            self.calculation_methods) + ' as String'

        # fundamental frequency calculation
        self.F_0 = pp.get_fundamental_frequency(self.acoustic, framerate, mode=self.F_0_determination_method,
                                                centroid_periodogram_db_treshold=self.centroid_periodogram_db_treshold,
                                                plot=False)

        # filter acoustic signal around the fundamental frequency
        self.filtered_acoustic = pp.filter_butterworth_bandpass(self.acoustic, self.F_0, framerate,
                                                                self.F_0_bandpass_filter_range,
                                                                order=self.filter_order)

        # where starts the phonation
        self.r_0 = pp.Onset_detection_binary_crit(self.time_array, self.filtered_acoustic, framerate,
                                                  duration_hann_window=self.freq_crit, energy_crit=self.energy_crit,
                                                  freq_crit=self.freq_crit)

        # obtain oscilation peaks
        self.peaks = \
            scipy.signal.find_peaks(self.filtered_acoustic, height=self.filtered_acoustic[self.r_0], threshold=None,
                                    distance=None,
                                    prominence=None, width=None, wlen=None, rel_height=0.5, plateau_size=None)[0]

        # Smooth the cycle peaks  -> outlier avoidance
        # TODO: make it an option?
        rolling_mean_peaks = pp.get_rolling_mean(self.filtered_acoustic[self.peaks], window=self.cma_peaks_window,
                                                 mode='same')

        self.r_sat = np.where(rolling_mean_peaks >= self.r_sat_consideration_crit * np.max(rolling_mean_peaks))[0][0]

        # only consider peaks for function fitting from begin of the onset up to the maximum aplitude of the phonation
        self.fitting_peaks = self.peaks[:self.r_sat + 1]

        # searches for the first peak within the fitting peaks which are above a threshold
        if self.first_significant_peak_option == 'on':
            first_sig_peaks = \
                scipy.signal.find_peaks(self.filtered_acoustic[self.fitting_peaks],
                                        height=np.max(self.filtered_acoustic) * self.first_significant_peak_crit)[0]

            if len(first_sig_peaks) != 0:
                self.first_significant_peak = self.fitting_peaks[first_sig_peaks[0]]
                self.fitting_peaks = self.fitting_peaks[:first_sig_peaks[0] + 1]
            # if there is no peak e.g. the peaks are rising monotonously the fitting peak is set as first significant peak
            else:
                self.first_significant_peak = self.fitting_peaks[-1]

        # this is where the magic happens
        if method == 'kunduk':
            self.kunduk_method(self.time_array, self.acoustic, self.filtered_acoustic, self.fitting_peaks,
                               fitting_mode=self.kunduk_calculation_method)
        else:
            self.mergel_method(self.time_array, self.acoustic, self.filtered_acoustic, self.fitting_peaks,
                               fitting_mode=self.mergel_calculation_method)

        #Remove the silence added at the beginning of this function.
        self.onset_label = self.onset_label[int(framerate/2):int(len(self.onset_label)-(framerate/2))]
    
    #-------------------------------------------------------------------------------------------------------------
        
    def kunduk_method(self, time, acoustic, filtered_acoustic, peaks, fitting_mode):
        """
        calculates a variant of the vocal onset time with fitting polynom of n degree(standard 4) or spline function
        saves the Onset Time, Relativ Onset Time, Onset Label, and the cut out signal which are returnable with the
        appropriate functions


        -estimate average cycle lengths for support point
        -include support point one cycle length before first peak value
        -fit a nth degree polynom or a spline function to the to the choosen fitting_peaks
            when fitting a polynominal fct:
                10 times the weights for the support and maximum point to
                force polynom to be as close to these points ( no over-/ undershoot)
        -"self.begin_onset" of maximum polynom value is  begin steady state
        -"self.steady_state" of maximum polynom value is begin onset process
        -includes "self.considered_cycles" times the estimiated cycle length after begin steady state as the steady state phase
        -includes same time as the pre onset phase

       :param time: time array
       :type time: np.ndarray
       :param acoustic: raw signal
       :type acoustic: np.array
       :param filtered_acoustic: filtered signal around F_0
       :type filtered_acoustic: ndarray
       :param peaks: oscilation peaks from the begin of the onset process up to the maximum amplitude peak
       :type peaks: 'ndarray'
       :param fitting_mode:defines what function will be fitted -> Options: ['bspline','cspline','poly']
       :type fitting_mode: 'string'
       :param plot: switch to disable the plot option
       :type plot: Boolean
       """
        # throw assert if invalid calculation method occours
        assert fitting_mode in self.kunduk_calculation_methods, "Wrong method! Use one of the following methods: " + ' '.join(
            self.kunduk_calculation_methods) + ' as String'
        assert self.first_significant_peak_option in self.toggle_options, "Wrong string! Use one of the following options: " + ' '.join(
            self.toggle_options) + ' as String'

        old_peaks = peaks

        # for seeting the support point when fitting a polynom and additional to extract the steady state phase and pre onset phase from the signal
        self.mean_dist_peaks = int(np.mean(peaks[1:] - peaks[:-1]))
        # insert the support point for polynom stability
        self.steps_evaluation_points = np.insert(time[peaks], 0, time[peaks[0] - self.mean_dist_peaks])
        self.filtered_signal_evaluation_points = np.insert(filtered_acoustic[peaks], 0, 0)
        # cut out the time points for polynom fitting

        if peaks[0] < self.mean_dist_peaks:
            self.steps_evaluation = time[0:peaks[-1]]
        else:
            self.steps_evaluation = time[peaks[0] - self.mean_dist_peaks:peaks[-1]]
        # mode selection of the fitting methods
        if fitting_mode == 'bspline':
            t, c, k = scipy.interpolate.splrep(self.steps_evaluation_points, self.filtered_signal_evaluation_points,
                                               s=0, k=4)
            poly = scipy.interpolate.BSpline(t, c, k)
            self.evaluated_poly = poly(self.steps_evaluation)
        elif fitting_mode == 'cspline':
            poly = scipy.interpolate.CubicSpline(self.steps_evaluation_points, self.filtered_signal_evaluation_points,
                                                 axis=0,
                                                 bc_type='not-a-knot', extrapolate=None)
            self.evaluated_poly = poly(self.steps_evaluation)
        elif fitting_mode == 'poly':
            # weight the first and the last point for polynom stability -> prevent over and undershoot of poly
            # TODO: make weights adjustable?
            weights = np.ones(len(self.filtered_signal_evaluation_points))
            weights[0] = 10
            weights[-1] = 10
            poly = np.polyfit(self.steps_evaluation_points, self.filtered_signal_evaluation_points, self.poly_degree,
                              rcond=None,
                              full=False, w=weights, cov=False)
            self.evaluated_poly = np.zeros(len(self.steps_evaluation))
            for i, step in enumerate(self.steps_evaluation):
                for n, a in enumerate(poly[::-1]):
                    self.evaluated_poly[i] += (step ** n) * a

        self.steady_state_poly = next(
            x for x, val in enumerate(self.evaluated_poly)
            if val >= self.steady_state_requirement * np.max(self.evaluated_poly))
        self.begin_onset_poly = next(
            x for x, val in enumerate(self.evaluated_poly)
            if val >= self.begin_onset * np.max(self.evaluated_poly))

        # since the polynom gets only calculated on a segment of the signal -> need for the true indices

        self.begin_steady_state_index = np.where(time == self.steps_evaluation[self.steady_state_poly])[0][0]
        self.begin_onset_index = np.where(time == self.steps_evaluation[self.begin_onset_poly])[0][0]

        # label Signal -> for deep learning stuff

        if self.first_significant_peak_option == 'off':
            self.onset_label = util.get_onset_label(acoustic, self.r_0, self.peaks[self.r_sat],
                                                    self.mean_dist_peaks, self.considered_cycles)
        else:
            self.onset_label = util.get_onset_label(acoustic, self.r_0, self.first_significant_peak,
                                                    self.mean_dist_peaks, self.considered_cycles)

        self.norm_time = util.get_relativ_time(self.begin_onset_index, self.begin_steady_state_index,
                                               self.mean_dist_peaks, self.considered_cycles)

        # onset time
        self.time = time[self.begin_steady_state_index] - time[self.begin_onset_index]


    def mergel_method(self, time, acoustic, filtered_acoustic, peaks, fitting_mode):
        """
       calculates a variant of the vocal onset time by fitting a exponential saturation function
       either one parameters will be fitted ('a') or three parameters will be fitted ('a', 'r_0' , 'r_sat')
       saves the Onset Time, Relativ Onset Time, Onset Label, and the cut out signal which are returnable with the
       appropriate functions

       -fits the oscilation peaks from the start of the onset up to the maximum amplitude peaks to a
                exponential saturation function
        -x% of maximum polynom value is  begin steady state
        -y% of maximum polynom value is begin onset process
        -estimate average cycle lengths steady state and pre onset extraction
       -includes 5 times the estimiated cycle length after begin steady state as the steady state phase
       -includes same time as the pre onset phase


       :param time: time array
       :type time: np.ndarray
       :param acoustic: raw signal
       :type acoustic: np.array
       :param filtered_acoustic: filtered signal around F_0
       :type filtered_acoustic: ndarray
       :param peaks: oscilation peaks from the begin of the onset process up to the maximum amplitude peak
       :type peaks: 'ndarray'
       :param fitting_mode: defines how many free parameters will be fitted -> Options:
                                                ['one_free_parameter','three_free_parameter']
       :type fitting_mode: 'string'
       :param plot: switch to disable the plot option
       :type plot: Boolean
       """
        # throw assert if invalid calculation method occours
        assert fitting_mode in self.mergel_calculation_methods, "Wrong method! Use one of the following methods: " + ' '.join(
            self.mergel_calculation_methods) + ' as String'

        if fitting_mode == 'fit: a':

            param_bounds = ([1], [1000])
            popt, pcov = scipy.optimize.curve_fit(self.__hopf_burification_fct, time[peaks] - time[peaks[0]],
                                                  filtered_acoustic[peaks], bounds=param_bounds)
            self.evaluated_fct = self.__hopf_burification_fct(time - time[peaks[0]], popt[0])

        elif fitting_mode == 'fit: a, r_0, r_sat':

            param_bounds = ([1, 0, 0], [1000, 2 * filtered_acoustic[self.r_0], np.max(filtered_acoustic)])

            popt, pcov = scipy.optimize.curve_fit(self.__hopf_burification_fct_three_free_parameters,
                                                  time[peaks] - time[peaks[0]], filtered_acoustic[peaks],
                                                  bounds=param_bounds,
                                                  p0=[1, filtered_acoustic[self.r_0],
                                                      filtered_acoustic[
                                                          self.first_significant_peak] if self.first_significant_peak_option == 'on' else
                                                      filtered_acoustic[peaks[self.r_sat]]])

            self.evaluated_fct = self.__hopf_burification_fct_three_free_parameters(time - time[peaks[0]], popt[0],
                                                                                    popt[1],
                                                                                    popt[2])

        # where does the onset and steady state starts, normaly the 1/a denotes the onset time, but we have to also
        # extract the signal
        self.begin_onset_index = next(x for x, val in enumerate(self.evaluated_fct)
                                      if val >= self.begin_onset_mergel * np.max(self.evaluated_fct))
        self.begin_steady_state_index = next(x for x, val in enumerate(self.evaluated_fct)
                                             if val >= self.steady_state_mergel * np.max(self.evaluated_fct))
        self.mean_dist_peaks = int(np.mean(peaks[1:] - peaks[:-1]))

        # label Signal -> for deep learning stuff
        if self.first_significant_peak_option == 'off':
            self.onset_label = util.get_onset_label(acoustic, self.r_0, self.peaks[self.r_sat],
                                                    self.mean_dist_peaks, self.considered_cycles)
        else:
            self.onset_label = util.get_onset_label(acoustic, self.r_0, self.first_significant_peak,
                                                    self.mean_dist_peaks, self.considered_cycles)

        self.norm_time = util.get_relativ_time(self.begin_onset_index, self.begin_steady_state_index,
                                               self.mean_dist_peaks, self.considered_cycles)

        # onset time
        self.time = time[self.begin_steady_state_index] - time[self.begin_onset_index]

    def __hopf_burification_fct_NM(self, time, a, r_0, r_sat, b):

        xi = (r_0 / r_sat) ** 2.0
        return r_0 * 1.0 / (np.sqrt((1.0 - xi) * math.e ** (-2.0 * a * time - b) + (xi)))

    def __hopf_burification_fct(self, x, a):
        """
        helper function
        exponential saturation function with a as the only free parameter
        r_0 & r_sat get defined before calling the function

        :param x: time array
        :type x: np.ndarray
        :param a: denotes the slope of the function
        :type a: float

        :returns the evaluated function:
        :rtype: float or ndarray
        """
        if self.first_significant_peak_option == 'on':
            xi = (self.filtered_acoustic[self.r_0] / self.filtered_acoustic[self.first_significant_peak]) ** 2.0
        else:
            xi = (self.filtered_acoustic[self.r_0] / self.filtered_acoustic[self.fitting_peaks[self.r_sat]]) ** 2.0

        return self.filtered_acoustic[self.r_0] * 1.0 / (np.sqrt((1.0 - xi) * math.e ** (-2.0 * a * x) + (xi)))

    def __hopf_burification_fct_three_free_parameters(self, x, a, r_0, r_sat):

        """
        helper function
        exponential saturation function with a, r_0 & r_sat as the free parameters

        :param x: time array
        :type x: np.ndarray
        :param a: denotes the slope of the function
        :type a: float
        :param r_0: denotes the amplitude value at which the phonation starts
        :type r_0: float
        :param r_sat: denotes the the maximum amplitude value
        :type r_sat: float

        :returns the evaluated function:
        :rtype: float or ndarray
        """
        xi = (r_0 / r_sat) ** 2.0
        return r_0 * 1.0 / (np.sqrt((1.0 - xi) * math.e ** (-2.0 * a * x) + (xi)))

    def set_new_hyperparameter_kunduk(self, dict):
        '''changes the hyperparameters for the kunduk calculation method'''
        self.kunduk_calculation_method = dict['kunduk_calculation_method']
        self.first_significant_peak_option = dict['first_significant_peak_option']
        self.reduce_signal = dict['reduce_signal']
        self.poly_degree = dict['poly_degree']
        self.begin_onset = dict['begin_onset']
        self.steady_state_requirement = dict['steady_state_requirement']
        self.first_significant_peak_crit = dict['first_significant_peak_crit']
        self.F_0_determination_method = dict['F_0_determination_method']
        self.centroid_periodogram_db_treshold = dict['centroid_periodogram_db_treshold']
        self.F_0_bandpass_filter_range = dict['F_0_bandpass_filter_range']
        self.filter_order = dict['filter_order']
        self.duration_hann_window = dict['duration_hann_window']
        self.energy_crit = dict['energy_crit']
        self.freq_crit = dict['freq_crit']
        self.r_sat_consideration_crit = dict['r_sat_consideration_crit']
        self.cma_peaks_window = dict['cma_peaks_window']
        self.considered_cycles = dict['considered_cycles']

    def set_new_hyperparameter_mergel(self, dict):
        '''changes the hyperparameters for the mergel calculation method'''
        self.mergel_calculation_method = dict['mergel_calculation_method']
        self.first_significant_peak_option = dict['first_significant_peak_option']
        self.first_significant_peak_crit = dict['first_significant_peak_crit']
        self.reduce_signal = dict['reduce_signal']
        self.begin_onset_mergel = dict['begin_onset_mergel']
        self.steady_state_mergel = dict['steady_state_mergel']
        self.F_0_determination_method = dict['F_0_determination_method']
        self.centroid_periodogram_db_treshold = dict['centroid_periodogram_db_treshold']
        self.F_0_bandpass_filter_range = dict['F_0_bandpass_filter_range']
        self.filter_order = dict['filter_order']
        self.duration_hann_window = dict['duration_hann_window']
        self.energy_crit = dict['energy_crit']
        self.freq_crit = dict['freq_crit']
        self.r_sat_consideration_crit = dict['r_sat_consideration_crit']
        self.cma_peaks_window = dict['cma_peaks_window']
        self.considered_cycles = dict['considered_cycles']
#
# data = fl.load(r"..\Split_recordings\S001_VOMS_280417-03.h5")
# acoustic = data['data']['acoustic'].values
# framerate = data['info']['fs']
# time = data['data']['time'].values
# vot = VoiceOnsetTime(first_significant_peak_option='on', F_0_bandpass_filter_range = 0.1 )
# vot.compute(time,acoustic,framerate,plot=True, method = 'kunduk')
