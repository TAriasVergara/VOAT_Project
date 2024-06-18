import pandas as pd
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt


def hamming_moving_average_filter(steps, signal, f_0, filter_range,framerate, plot=False):
    """
    Helper fct for recursive_detrending_bandpass_filter
    Multiplies a hamming window with a lowpass filter

    :returns:  ndarray


    :param steps: time of the acoustic signal
    :type steps: ndarray
    :param signal: signal which should be filtered
    :type signal: ndarray
    :param f_0: fundamental frequency
    :type f_0: float
    :param filter_range: filter frequency in percent w.r.t. the fundamental frequency
    :type filter_range: float
        Example:: f_0 *(1.0 - filter_range)
    :param plot: switch if the calculation process should be plotted
    :type plot: boolean
    """
    # https: // tomroelandts.com / articles / how - to - create - a - simple - low -pass-filter
    fc = f_0 * (1 - filter_range) / (framerate / 2)

    b = fc * 0.9  # Transition band, as a fraction of the sampling rate (in (0, 0.5)).
    N = int(np.ceil((4 / b)))
    if not N % 2: N += 1  # Make sure that N is odd.
    n = np.arange(N)

    # Compute sinc filter.
    h = np.sinc(2 * fc * (n - (N - 1) / 2))

    # Compute hamming window.
    w = np.hamming(N)

    # Multiply sinc filter by window.
    h = h * w

    # Normalize to get unity gain.
    h = h / np.sum(h)
    s = np.convolve(signal, h, 'same')
    if plot == True:
        plt.title('signal filtered with hamming window moving average method')
        plt.plot(steps, s)
    return s

def recursive_detrending_bandpass_filter(time, signal, filter_range, framerate, f_0, plot=False):
    #TODO: decide if filtering should be done by simple firwin method or design the filter by hand
    """
    Performs a recursive detrending filtering as proposed in:
    Measures of Vocal Attack Time for Healthy Young Adults, Roark et al.

    :returns: ndarray


    :param time: time of the acoustic signal
    :type time: ndarray
    :param signal: signal which should be filtered
    :type signal: ndarray

    :param filter_range: filter frequency in percent w.r.t. the fundamental frequency
    :type filter_range: float
        Example:: f_0 *(1.0 +/- filter_range)
    :param framerate: sampling frequency
    :type framerate: float
    :param f_0: fundamental frequency
    :type f_0: float
    :param plot: switch if the calculation process should be plotted
    :type plot: boolean
    """
    taps = scipy.signal.firwin(1500, (1.0 + filter_range) * f_0, nyq=framerate / 2, pass_zero=True,
                               window='hamming', scale=False)

    filtered_signal_low = scipy.signal.filtfilt(taps, 1.0, signal)

    filtered_signal_low2 = hamming_moving_average_filter(time, filtered_signal_low, f_0, filter_range,framerate, plot=False)

    taps = scipy.signal.firwin(1500, (1.0 - filter_range) * f_0, nyq=framerate / 2, pass_zero=True,
                               window='hamming', scale=False)
    #filtered_egg_low2_1 = scipy.signal.filtfilt(taps, 1.0, signal)

    tmp = filtered_signal_low - filtered_signal_low2
    #tmp_1 = filtered_egg_low - filtered_egg_low2_1

    for i in range(4):
        filtered_egg_low2 = hamming_moving_average_filter(time, tmp, f_0, filter_range, framerate, plot=False)

        #taps = scipy.signal.firwin(1500, (1.0 - filter_range) * f_0, nyq=framerate / 2, pass_zero=True, window='hamming', scale=False)
        #filtered_egg_low2_1 = scipy.signal.filtfilt(taps, 1.0, tmp_1)

        tmp = filtered_signal_low - filtered_egg_low2

        #tmp_1 = filtered_egg_low - filtered_egg_low2_1

    tmp = normalize(tmp)

    #tmp_1 = normalize(tmp_1)
    if plot:
        plt.subplot(411)
        plt.title('extracted_egg')
        plt.plot(time, signal)
        plt.subplot(412)
        plt.title('Lowpass filtered egg signal 1.4*F_0')
        plt.plot(time, filtered_signal_low)
        plt.subplot(413)
        plt.title('Lowpass filtered egg signal with moving average hamm window 0.6*F_0')
        plt.plot(time, filtered_signal_low2)
        plt.subplot(414)
        plt.title('Bandpass filtered signal with recursive detrending method design the filter "by hand"')
        plt.plot(time, tmp)
        # plt.subplot(615)
        # plt.title('Bandpass filtered signal with recursive detrending method design the filterby scipy.firwin method ')
        # plt.plot(time, tmp_1)
        # plt.subplot(616)
        # plt.title('difference between the bandpass filtered signals ')
        # plt.plot(time, tmp - tmp_1)
        plt.show()

    return tmp

def filter_signal_firwin_bandpass(signal, lowcut, highcut, order, framerate):

    """
    Performs a bandpass filtering with a finite impulse response window method


    :returns:  pandas.dataframe

    :param signal: contains the signal
    :type signal: np.ndarray
    :param lowcut: lower frequency range for the bandpass filter
    :type lowcut: int
    :param highcut: higher frequency range for the bandpass filter
    :type highcut: int
    :param order: order of the filter
    :type order: int
    :param framerate: windowsize
    :type framerate: int

    """
    taps = scipy.signal.firwin(order, [lowcut, highcut], nyq=framerate / 2, pass_zero=False,
                               window='hamming', scale=False)

    filtered_signal = scipy.signal.filtfilt(taps, 1.0, signal)
    # filtered_signal = scipy.signal.lfilter(taps,1.0,signal)

    return filtered_signal

def normalize(signal, mode = '-1&1'):
    """
     normalizes the signal with either:
     1) scaling the signal between -1 and 1
     2) normalizes the signal by substracting the mean and dividing by the standard deviation

    :returns:  np.ndarray


    :param signal: contains a signal
    :type signal: np.ndarray
    :param mode: decide which normalization is performed
    :type mide: String


    """
    if mode == '-1&1':
        return  2 * (signal - np.min(signal)) / (np.max(signal) - np.min(signal)) - 1
    else:
        return (signal - np.mean(signal)) / np.std(signal)


def get_rolling_std(signal, window=100):
    """
    calculated a windowed standard deviation
    :returns:  pandas.dataframe

    :param signal: contains the audiosignal
    :type signal: np.ndarray
    :param window: windowsize
    :type window: int

    """

    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.window.rolling.Rolling.std.html
    s = pd.Series(signal)
    rolling_std = s.rolling(window).std()
    return rolling_std

def get_rolling_mean(signal, window=20, mode ='same'):

    """
    Calculates a moving average

    :returns:  nd.array


    :param signal: contains the audiosignal
    :type signal: np.ndarray
    :param window: windowsize
    :type window: int
    :param mode: calculation style see np.convolve method
    :type window: String
    """

    return np.convolve(signal, np.ones((window,)) / window, mode=mode)

def get_hilbert_transform(signal):

    """
    calculates envelope with hilbert transform
   :returns:  nd.array


   :param signal: contains the audiosignal
   :type signal: np.ndarray
   """
    return np.abs(scipy.signal.hilbert(signal))

def throw_out_nan(signal):
    """
    Removes Nan Values
   :returns:  nd.array


   :param signal: contains the audiosignal
   :type signal: np.ndarray
   """

    nan_array = np.isnan(signal)
    not_nan_array = ~ nan_array
    return signal[not_nan_array]

def freq_from_autocorr(sig, fs, plot):
    # https: // gist.github.com / endolith / 255291
    """
    Estimate frequency using autocorrelation

    :returns:  float


    :param sig: contains the audiosignal
    :type sig: np.ndarray
    :param fs: framerate
    :type fs: float
    """
    # Calculate autocorrelation and throw away the negative lags
    corr = scipy.signal.correlate(sig, sig, mode='full')
    corr = corr[len(corr) // 2:]



    # Find the first low point
    d = np.diff(corr)
    start = np.nonzero(d > 0)[0][0]

    # Find the next peak after the low point (other than 0 lag).  This bit is
    # not reliable for long signals, due to the desired peak occurring between
    # samples, and other peaks appearing higher.
    # Should use a weighting function to de-emphasize the peaks at longer lags.
    peak = np.argmax(corr[start:]) + start
    px, py = parabolic(corr, peak)

    if plot:
        plt.plot(corr)
        plt.plot(px, py, 'o',label='First maximum after 0 point should denote the fundamental frequency')
        plt.title('Debuging view:  Autocorrelation result to determine the fundamental frequency')
        plt.legend(loc='lower right')
        plt.show()

    return fs / px

def parabolic(f, x):
    """Quadratic interpolation for estimating the true position of an
    inter-sample maximum when nearby samples are known.
    f is a vector and x is an index for that vector.
    Returns (vx, vy), the coordinates of the vertex of a parabola that goes
    through point x and its two neighbors.
    Example:
    Defining a vector f with a local maximum at index 3 (= 6), find local
    maximum if points 2, 3, and 4 actually defined a parabola.
    In [3]: f = [2, 3, 1, 6, 4, 2, 3, 1]
    In [4]: parabolic(f, argmax(f))
    Out[4]: (3.2142857142857144, 6.1607142857142856)
    """
    xv = 1 / 2. * (f[x - 1] - f[x + 1]) / (f[x - 1] - 2 * f[x] + f[x + 1]) + x
    yv = f[x] - 1 / 4. * (f[x - 1] - f[x + 1]) * (xv - x)
    return (xv, yv)

def get_fft(signal,framerate):

    """
    calculates the fft of a signal
    returns the frequency & fourier coefficient array
    :returns:  np.ndarray , np.ndarray


    :param spf: contains the framerate
    :type spf: wave.Wave_read object
    :param signal: Audio signal
    :type signal: np.ndarray
    """

    Fs = framerate  # sample rate
    N = len(signal)  # total points in signal

    Y_k = np.fft.fft(signal)[0:int(N / 2)] / N  # FFT function from numpy
    Y_k[1:] = 2 * Y_k[1:]  # need to take the single-sided spectrum only
    Pxx = np.abs(Y_k)  # be sure to get rid of imaginary part

    # f = Fs * np.arange((N / 2)) / N;  # frequency vector
    f = np.linspace(0, Fs // 2, N // 2)
    return f, Pxx


def get_fundamental_frequency(signal, framerate, F_0_peak_consider_treshold=0.05,centroid_periodogram_db_treshold=-4, mode='correlation', plot=True):

    """
    calculates the fundamentalfrequency from either the autocorrelation method or find the first peak in the frequency
    domain which exceeds F_0_peak_consider_treshold

    returns the fundamental frequency
    :returns:  float

    :param signal: Audio signal
    :type signal: np.ndarray
    :param F_0_peak_consider_treshold: for F_0 frequency determination method. only frequency coefficients above
    threshold are considered
    :type F_0_peak_consider_treshold: float
    :param mode: specifies the calculation method used to calculate the fundamental frequency
                 options: ['fft_peak', 'correlation', 'centroid_periodogram']
    :type mode: string
    :param plot: print both of the F_0 Values
    :type plot: Boolean
    """

    mode_list = ['fft_peak', 'correlation', 'centroid_periodogram']
    assert mode  in mode_list, "Wrong method! Use one of the following methods" + ' '.join(mode_list) + ' as string'

    #calculation Method periodogram estimation

    f, Pxx_spec = scipy.signal.periodogram(signal, framerate, scaling='spectrum')
    #TODO: decide if go for db threshhold or maximum value
    db_treshold = centroid_periodogram_db_treshold
    Sqrt_Pxx_spec = np.sqrt(Pxx_spec)
    if int(centroid_periodogram_db_treshold) < 0:
        #candidates = np.where(Sqrt_Pxx_spec > 10 ** (db_treshold / 20) * Sqrt_Pxx_spec.max())[0]
        candidates = scipy.signal.find_peaks(Sqrt_Pxx_spec, height=10 ** (db_treshold / 20) * Sqrt_Pxx_spec.max(),
                                        threshold=None, distance=None,
                                        prominence=None, width=None,
                                        wlen=None,
                                        rel_height=0.5, plateau_size=None)[0]
    elif centroid_periodogram_db_treshold == 0:
        candidates = np.where(Sqrt_Pxx_spec == np.max(Sqrt_Pxx_spec))[0]
    denominator = 0
    devider = 0
    for candidate_index in candidates:
        denominator = denominator + (Sqrt_Pxx_spec[candidate_index] * f[candidate_index])
        devider = devider + Sqrt_Pxx_spec[candidate_index]

    centroid_freq = denominator / devider



    #get first significant peak in frequency domain
    frequencies, absfft_signal = get_fft(signal,framerate)

    peaks = scipy.signal.find_peaks(absfft_signal, height=np.max(absfft_signal) * F_0_peak_consider_treshold,
                                    threshold=None, distance=None,
                                    prominence=None, width=None,
                                    wlen=None,
                                    rel_height=0.5, plateau_size=None)[0]

    freq = frequencies[peaks[0]]


    #freq from autocorrelation
    f_0 = freq_from_autocorr(signal, framerate, plot=False)

    if plot:
        print('F_0 value: ' + str(freq) + ' with first considerable peak (5% of max) in frequency spectrum method')
        print('F_0 value: ' + str(f_0) + ' with autocorrelation method')
        print('F_0 value: ' + str(centroid_freq) + ' with centroid of periodogram method')

        if mode == 'correlation':
            f_0 = freq_from_autocorr(signal, framerate, plot=plot)

        elif mode == 'fft_peak':
            plt.plot(frequencies, absfft_signal)
            plt.plot(frequencies[peaks[0]], absfft_signal[peaks[0]], 'o', label='fundamental frequency')
            plt.legend(loc='lower right')
            plt.title('first peak over 5% from max frequency is considered the fundamental freq')
            plt.xlabel('frequency')
            plt.show()

        elif mode == 'centroid_periodogram':
            plt.plot(f, Sqrt_Pxx_spec)
            plt.plot(f[candidates], Sqrt_Pxx_spec[candidates], 'o', label = 'fundamental frequency candidates')
            plt.hlines(10 ** (db_treshold / 20) *Sqrt_Pxx_spec.max(), f[0], f[-1], colors='k', linestyles='solid',
                       label='-4 dB threshold')
            plt.legend(loc = 'upper right')
            #plt.title('Debug view: Estimation of fundamental frequency with periodogram')
            plt.title('Estimation of $f_0$  with an periodogram, $f_0$  ='+ "{:.1f}".format(centroid_freq))

            plt.show()

    if mode == 'correlation':
        return f_0
    elif mode == 'fft_peak':
        return freq
    elif mode == 'centroid_periodogram':
        return centroid_freq


def filter_butterworth_bandpass(signal, freq, framerate, filter_range, order=4):
    """
    filters the with a bandpass filter around the fundamental frequency F_0
    returns the filtered signal
    :returns:  np.ndarray

    :param signal: Audio signal
    :type signal: np.ndarray
    :param spf: contains the framerate
    :type spf: wave.Wave_read object
    :param freq: fundamental frequency F_0
    :type freq: float
    :param lowcut_range: lowest frequency percentage which should be preserved
    :type lowcut_range: float
    :param highcut_range: highest frequency percentage which should be preserved
    :type highcut_range: float
    """
    lowcut_range = 1.0 - filter_range
    highcut_range = 1.0 + filter_range
    lowcut = lowcut_range * freq
    highcut = highcut_range * freq
    b, a = butter_bandpass(lowcut, highcut, framerate, order)

    filtered_signal = scipy.signal.filtfilt(b, a, signal)

    return filtered_signal

def butter_bandpass(lowcut, highcut, fs, order):
    """
    helper method for   filter_signal_bandpass
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(order, [low, high], btype='band', analog=False)
    return b, a

def Onset_detection_binary_crit(steps, signal, framerate, duration_hann_window = 0.025, energy_crit=0.15, freq_crit=0.15):
    """
    binary decision process, which defines a starting point of the onset process
    the first oscillation peak which both reaches a certain percentage value of the maximum energy
    and is in within a percentage value of the median frequency the gets to be chosen as the starting point

    if the signal as too noisy and a stable median frequency can not be found the exception will be catched and
    the median frequency will be replaced by the centroid frequency of the periodogram

    returns the index of the starting point
    :returns:  int


    :param steps: contains the time information of the signal
    :type steps: np.ndarray
    :param signal: Audio signal
    :type signal: np.ndarray
    :param framerate: sampling frequency
    :type framerate: float
    :param duration_hann_window: duration of the hann window in seconds, used as a preprocessing lowpass filter
    :type duration_hann_window: float
    :param energy_crit: first criteria for the starting point, the starting point has to have at least a certain
                        percentage value of the maximum energy peak
    :type energy_crit: float
    :param freq_crit: second criteria for the starting, the frequency between two oscillation peaks has to be around
                      a certain percentage value of the median frequency
    :type freq_crit: float
    """

    #lowpass filtering and calculating the energy contour of the signal
    dt_signal = 1/framerate
    size_hann_window = int(duration_hann_window / dt_signal)
    hann_window = scipy.signal.windows.hann(size_hann_window)
    local_energy = np.convolve(signal ** 2, hann_window, mode='same')

    #peak finding for determination of the frequency stability
    peaks = scipy.signal.find_peaks(signal, height=np.max(signal) * 0.01,
                                    threshold=None, distance=None,
                                    prominence=None, width=None,
                                    wlen=None,
                                    rel_height=0.5, plateau_size=None)[0]

    median_freq = 1 / np.median(steps[peaks][1:] - steps[peaks][:-1])
    freq_peaks = 1 / (steps[peaks][1:] - steps[peaks][:-1])

    #which points fullfil the critirea of frequency stability
    freq_crit_points = np.where((freq_peaks > median_freq * (1-freq_crit)) & (freq_peaks < median_freq * (1+freq_crit)))[0]
    #whicht point fullfil the energy criteria
    energy_crit_points = np.where(local_energy[peaks[freq_crit_points]] > energy_crit * np.max(local_energy))[0]
    #if no stable median frequency can be obtained, replace the median freuqncy by the centroid frequency of the
    #periodogram
    try:
        start_point_phonation_index = peaks[freq_crit_points[energy_crit_points[0]]]

    except Exception:
        median_freq = get_fundamental_frequency(signal,framerate,mode='centroid_periodogram', plot = False)
        freq_crit_points = np.where((freq_peaks > median_freq * (1 - freq_crit)) & (freq_peaks < median_freq * (1 + freq_crit)))[0]
        print('______Catched an Exception______')
        print('______signal seems to be too distorted to get a usable median frequency______')
        print('______Instead of using the freq obtained from the median distance between signal peaks______')
        print('______F_0 gets determined from the get_fundamental_frequency fct______')

        energy_crit_points = np.where(local_energy[peaks[freq_crit_points]] > energy_crit * np.max(local_energy))[0]
        start_point_phonation_index = peaks[freq_crit_points[energy_crit_points[0]]]

    return start_point_phonation_index
