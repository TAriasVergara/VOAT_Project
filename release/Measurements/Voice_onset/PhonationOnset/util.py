import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
#TODO: documentation
def get_onset_label(signal, start_phonation, start_steady_state, distance_cycles, considered_cycles):

    """
    returns the relative onset time, w.r.t. to the amount of considered_cycles

    :returns:  ndarray


    :param start_phonation: index at which the onset process starts
    :type start_phonation: int
    :param start_steady_state:  index at which the onset process ends
    :type start_steady_state: int
    :param distance_cycles: average distance between oscillation cycles
    :type distance_cycles: int
        Example:: f_0 *(1.0 - filter_range)
    :param considered_cycles: how many oscillation cycles should be included into the onset_label
    :type considered_cycles: int

    """

    onset_label = np.zeros(len(signal))
    onset_label[start_phonation - considered_cycles *
                distance_cycles:start_steady_state + considered_cycles * distance_cycles] = 1
    return onset_label

def get_relativ_time(start_phonation, start_steady_state, distance_cycles, considered_cycles):
    """
    one hot encodes the onset process of the given signal, w.r.t. to the amount of considered_cycles

    :returns:  ndarray

    :param signal: the signal which should be one hot encoded
    :type signal: ndarray
    :param start_phonation: index at which the onset process starts
    :type start_phonation: int
    :param start_steady_state:  index at which the onset process ends
    :type start_steady_state: int
    :param distance_cycles: average distance between oscillation cycles
    :type distance_cycles: int
        Example:: f_0 *(1.0 - filter_range)
    :param considered_cycles: how many oscillation cycles should be included into the onset_label
    :type considered_cycles: int

    """

    if considered_cycles == 0:
        return 1.0
    else:
        norm_duration = np.linspace(0, 1, (start_steady_state + considered_cycles * distance_cycles) -
                                    (start_phonation - considered_cycles * distance_cycles))
        norm_pre_phonation_time = norm_duration[:considered_cycles * distance_cycles][-1]
        norm_onset_time = np.max(norm_duration) - 2 * norm_pre_phonation_time
        return norm_onset_time
