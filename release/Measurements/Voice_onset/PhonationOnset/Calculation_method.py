from abc import ABC, abstractmethod

class Calculation_method(ABC):

    @abstractmethod
    def compute(self):pass
    # @abstractmethod
    # def get_time(self):pass
    # @abstractmethod
    # def get_relative_time(self):pass
    # @abstractmethod
    # def get_onset_label(self):pass
    # @abstractmethod
    # def get_cut_signal(self):pass