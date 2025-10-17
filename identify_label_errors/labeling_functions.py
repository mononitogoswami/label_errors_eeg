####################################################
#  Labeling functions to capture domain knowledge
####################################################
"""
Neurological domain knowledge-driven labeling functions for EEG analysis.

This module implements expert knowledge as computational labeling functions
that can automatically classify EEG signal patterns based on clinical criteria
such as baseline amplitude, spike characteristics, and signal distribution patterns.
"""

from typing import List, Tuple, Optional, Union
import numpy as np
from scipy.stats.mstats import winsorize
from sklearn.mixture import GaussianMixture
from scipy.signal import find_peaks
from .utils import fit_robust_line, fill_nans_with_zeros, Config


class LabelingFunctions(object):
    """
    Domain knowledge-driven labeling functions for EEG signal classification.

    This class implements 20+ specialized labeling functions that capture
    neurological domain expertise for automated EEG pattern recognition.
    Each function applies specific clinical criteria to classify signals into
    categories: Normal, Suppressed, Suppressed with Ictal, Burst Suppression.

    The approach uses:
    - Baseline trend analysis via robust linear regression
    - Peak detection with prominence thresholds
    - Gaussian Mixture Models for bimodal pattern detection
    - Signal amplitude distribution analysis
    - Temporal pattern characterization

    Attributes:
        eeg (np.ndarray): Processed EEG signal data
        baseline_exists (bool): Whether baseline trend analysis succeeded
        slope (float): Slope of fitted baseline trend
        average_eeg_baseline (float): Mean baseline amplitude
        peaks (np.ndarray): Indices of detected signal peaks
        prominences (np.ndarray): Peak prominence values above baseline
        n_peaks_ha (int): Count of high amplitude peaks (>10mV)
        n_peaks_la (int): Count of low amplitude peaks (5-10mV)
        means (List[float]): Gaussian mixture component means
        weights (List[float]): Gaussian mixture component weights

    Example:
        >>> eeg_signal = np.random.randn(1000) + 5  # Simulated EEG
        >>> lf = LabelingFunctions(eeg_signal)
        >>> vote_vector = lf.get_vote_vector()
        >>> print(f"Labeling function outputs: {vote_vector}")
    """

    def __init__(self,
                 eeg: np.ndarray,
                 config_file_path: str = 'config.yaml') -> None:
        """
        Initialize labeling functions with EEG data and configuration.

        Args:
            eeg: Raw EEG signal data as numpy array
            config_file_path: Path to YAML configuration file containing
                            labeling function parameters and thresholds

        Raises:
            FileNotFoundError: If config file is not found
            ValueError: If EEG data is empty or invalid
        """
        # Input validation
        if not isinstance(eeg, (np.ndarray, list)) or len(eeg) == 0:
            raise ValueError("EEG data must be non-empty numpy array or list")

        # Assign inputs to self
        self.read_inputs_into_self(config_file_path)

        # Read, process and mask aEEG data
        self.eeg, self.filled_nans = fill_nans_with_zeros(eeg)
        self.length_eeg = len(self.eeg)

        if self.length_eeg < 100:  # Minimum signal length for reliable analysis
            raise ValueError(f"EEG signal too short ({self.length_eeg} samples). Minimum 100 samples required.")

        self.process_eeg()

        # Characterize the spikes in the aEEG data
        self.characterize_spikes_in_data()

        if self.verbose:
            self.print_statistics()
    
    def process_eeg(self) -> None:
        """
        Process and clean EEG signal data.

        Applies signal processing steps including amplitude clipping,
        artifact removal via winsorization, and robust baseline fitting.
        """
        # Clip amplitudes above 50 mV (typical artifact threshold for aEEG)
        self.filled_nans = np.logical_and(self.eeg < 50, self.filled_nans)
        
        if self.filled_nans is not None:
            self.eeg = self.eeg[~self.filled_nans]  # Keep only not NaN indices
        else:
            self.eeg = self.eeg
        
        self.eeg = winsorize(a=self.eeg, limits=[0.01, 0.01], inplace=False)  # To remove artefacts

        # Fit a line to the EEG signal
        self.slope, self.intial_eeg_baseline, _, self.y_pred = fit_robust_line(self.eeg)
        self.final_eeg_baseline = float(self.y_pred[-1])
        self.average_eeg_baseline = np.sum(self.y_pred) / len(self.y_pred)
        self.peaks, _ = find_peaks(self.eeg.reshape((-1,)),prominence=2,width=1)
        self.prominences = self.eeg[self.peaks].reshape((-1,)) - self.y_pred.reshape((-1,))[self.peaks]
    
    def read_inputs_into_self(self, config_file_path):
        # Read config file
        args = Config(config_file_path=config_file_path).parse()
        self.description = args['expert_description']
        self.explain = args['explain_decisions']
        self.lf_params = args['labeling_function_parameters']
        self.verbose = args['verbose']

        self.abstain = args['label_values']['abstain'] 
        self.normal = args['label_values']['normal']
        self.suppressed = args['label_values']['suppressed']
        self.suppressed_with_ictal = args['label_values']['suppressed_with_ictal']
        self.burst_suppression = args['label_values']['burst_suppression']
  
    def characterize_spikes_in_data(self):
        # Initializations 
        self.no_baseline = False
        self.means = []
        self.weights = []
        self.bic = []

        # Number of high and low amplitude peaks
        if len(self.peaks) > 0:  # There may be no peaks at all
            self.n_peaks_vha = len(self.peaks[self.prominences > self.lf_params['eeg_high_15']])
            self.n_peaks_ha = len(self.peaks[self.prominences > self.lf_params['eeg_high_10']])
            self.n_peaks_la = len(self.peaks[self.prominences > self.lf_params['eeg_high_5']]) - self.n_peaks_ha
        else:
            self.n_peaks_vha, self.n_peaks_ha, self.n_peaks_la = 0

        self.many_high_amp_spikes = self.n_peaks_ha > self.lf_params['n_high_amplitude_peaks_per_hour']
        self.many_low_amp_spikes = self.n_peaks_la > self.lf_params['n_high_amplitude_peaks_per_hour']
        self.low_baseline = self.average_eeg_baseline < self.lf_params['eeg_low_2']
        self.dur_low_amplitude_eeg = len(self.eeg[self.eeg < self.lf_params['near_zero']])

        # Fit Gaussian Mixtures
        for n_components in [1, 2]:
            obj = GaussianMixture(n_components=n_components)
            obj.fit(self.eeg)
            self.bic.append(obj.bic(self.eeg))
            self.weights.append(obj.weights_.squeeze())
            self.means.append(obj.means_.squeeze())

    def print_decision_factors(self):
        """
        Print decision making factors
        """
        print('\t #############')
        print(f'\t Slope: {np.around(float(self.slope), 3)}  y-intercept: {np.around(float(self.intial_eeg_baseline), 3)}')
        print(f'\t Average eeg baseline: {np.around(self.average_eeg_baseline, 3)}')
        print(f'\t NaN time period: {np.sum(self.filled_nans)}')
        print(f'\t Peaks (> 5mV): {len(self.peaks)}')
        print(f'\t 1-component GMM: means = {self.means[0]} | weights = {self.weights[0]} BIC = {self.bic[0]}')
        print(f'\t 2-component GMM: means = {self.means[1]} | weights = {self.weights[1]} BIC = {self.bic[1]}')
        print(f"\t Number of high amplitude (> {self.lf_params['eeg_high_10']} mV) peaks {self.n_peaks_ha}")
        print(f"\t Number of low amplitude ({self.lf_params['eeg_high_5']} < _ < 10 mV) peaks {self.n_peaks_la }")
        print(f'\t Duration of near-zero aeeg amplitude (< 1mV): {self.dur_low_amplitude_eeg}')
        print(f'\t Not-NaNs eeg signal length: {len(self.eeg)} \n\t Minimum eeg value: {min(self.eeg)}')
        print('\t #############')

    def high_aeeg_baseline_normal(self, threshold_eeg_high=10):
        """
        high shaggy aeeg baseline constantly at an amplitude of around 10-20 mV, then normal eeg.
        """
        if ((self.average_eeg_baseline >= threshold_eeg_high) and (not self.no_baseline)):
            return self.normal
        else:
            return self.abstain

    def unimodal_aeeg_normal(self):
        if (min(self.weights[1]) < 0.05) and (
                self.means[0] >
                self.lf_params['eeg_low_2']):  # Unimodal distribution
            return self.normal
        else:
            return self.abstain

    def unimodal_aeeg_suppressed(self):
        if (min(self.weights[1]) < 0.05) and (
                self.means[0] <
                self.lf_params['eeg_low_2']):  # Unimodal distribution
            return self.suppressed
        else:
            return self.abstain

    def bimodal_aeeg_suppressed(self):
        if (min(self.weights[1]) > 0.05) and np.max(
                self.means[1]) < self.lf_params['eeg_low_2']:
            return self.suppressed
        else:
            return self.abstain

    def bimodal_aeeg_suppressed_with_ictal(self):
        if (min(self.weights[1]) > 0.05) and (np.min(
                self.means[1]) < self.lf_params['eeg_low_2']) and (np.max(
                    self.means[1]) > self.lf_params['eeg_high_5']):
            return self.suppressed_with_ictal
        else:
            return self.abstain

    def bimodal_aeeg_burst_suppression(self):
        if (min(self.weights[1]) > 0.05) and (np.min(
                self.means[1]) < self.lf_params['eeg_low_2']) and (np.max(
                    self.means[1]) < self.lf_params['eeg_high_5']):
            return self.burst_suppression
        else:
            return self.abstain

    def bimodal_aeeg_normal(self):
        if (min(self.weights[1]) > 0.05) and np.min(
                self.means[1]) > self.lf_params['eeg_low_2']:
            return self.normal
        else:
            return self.abstain

    def bimodal_aeeg(self):
        if min(self.weights[1]) > 0.05:
            if np.max(self.means[1]) < self.lf_params['eeg_low_2']:
                return self.suppressed
            elif (np.min(self.means[1]) < self.lf_params['eeg_low_2']) and (
                    np.max(self.means[1]) > self.lf_params['eeg_high_5']):
                return self.suppressed_with_ictal
            elif (np.min(self.means[1]) < self.lf_params['eeg_low_2']) and (
                    np.max(self.means[1]) < self.lf_params['eeg_high_5']):
                return self.burst_suppression
            elif np.min(self.means[1]) > self.lf_params['eeg_low_2']:
                return self.normal
            else:
                return self.abstain
        else:  # Unimodal distribution
            if self.means[0] < self.lf_params['eeg_low_2']:
                return self.suppressed
            if self.means[0] > self.lf_params['eeg_low_2']:
                return self.normal
            else:
                return self.abstain

    def aeeg_NOT_near_zero_normal(self):
        if (np.sum(self.eeg <= self.lf_params['near_zero']) <
                self.lf_params['near_zero_duration_tol']):
            return self.normal
        else:
            return self.abstain

    def very_spiky_aeeg_suppressed_with_ictal(self):
        """
        aEEF having spikes having > 15 mV more than once every minutes on an average is most probably ictal.
        """
        if self.n_peaks_vha > self.lf_params[
                'n_high_amplitude_peaks_per_hour']:
            return self.suppressed_with_ictal
        else:
            return self.abstain

    def well_separated_aeeg_modes_suppressed_with_ictal(self):
        """
        If aeeg values are well separated, i.e. their distribution has two peaks separated by atleast 4mV, 
        then the aeeg is more likey to be Supressed with ictal
        """
        if abs(self.means[1][0] -
               self.means[1][1]) > self.lf_params['min_separation'] and min(
                   self.weights[1]) > 0.05:
            return self.suppressed_with_ictal
        else:
            return self.abstain

    def low_baseline_suppressed_with_ictal(self):
        if (not self.no_baseline) and (self.low_baseline) and (
                self.many_high_amp_spikes):
            return self.suppressed_with_ictal
        else:
            return self.abstain

    def low_baseline_burst_suppression(self):
        if (not self.no_baseline) and (self.low_baseline) and (
                self.many_low_amp_spikes):
            return self.suppressed_with_ictal
        else:
            return self.abstain

    def low_baseline_suppressed(self):
        if (not self.no_baseline) and (self.low_baseline) and (
                not self.many_high_amp_spikes) and (
                    not self.many_low_amp_spikes):
            return self.suppressed
        else:
            return self.abstain

    def low_baseline_aeeg(self):
        if self.no_baseline:
            return self.abstain
        if self.low_baseline:
            if self.many_high_amp_spikes:
                return self.suppressed_with_ictal
            elif self.many_low_amp_spikes:
                return self.burst_suppression
            else:
                return self.suppressed
        else:
            return self.abstain

    def high_baseline_infrequent_drops_normal(self):
        if (not self.no_baseline) and (not self.low_baseline) and (
                self.dur_low_amplitude_eeg <=
                self.lf_params['near_zero_duration_tol']):
            return self.normal
        else:
            return self.abstain

    def high_baseline_frequent_drops_suppressed_with_ictal(self):
        if (not self.no_baseline) and (not self.low_baseline) and (
                self.dur_low_amplitude_eeg >
                self.lf_params['near_zero_duration_tol']
        ) and (self.bimodal_aeeg_burst_suppression() != self.burst_suppression):
            return self.suppressed_with_ictal
        else:
            return self.abstain

    def high_baseline_frequent_drops_burst_suppression(self):
        if (not self.no_baseline) and (not self.low_baseline) and (
                self.dur_low_amplitude_eeg >
                self.lf_params['near_zero_duration_tol']
        ) and (self.bimodal_aeeg_burst_suppression() == self.burst_suppression):
            return self.burst_suppression
        else:
            return self.abstain

    def high_baseline_frequent_drops(self):
        if self.no_baseline:
            return self.abstain
        if not self.low_baseline:
            if self.dur_low_amplitude_eeg <= self.lf_params[
                    'near_zero_duration_tol']:
                return self.normal
            elif self.dur_low_amplitude_eeg > self.lf_params[
                    'near_zero_duration_tol'] and self.bimodal_aeeg_burst_suppression(
                    ) == self.burst_suppression:
                return self.burst_suppression
            else:
                return self.suppressed_with_ictal
        else:
            return self.abstain

    def get_vote_vector(self):
        return [
            self.bimodal_aeeg(),
            self.low_baseline_aeeg(),
            self.high_baseline_frequent_drops(),
            self.very_spiky_aeeg_suppressed_with_ictal(),
            self.well_separated_aeeg_modes_suppressed_with_ictal(),
            self.aeeg_NOT_near_zero_normal(),
            self.high_aeeg_baseline_normal(threshold_eeg_high=4),
            self.high_aeeg_baseline_normal(threshold_eeg_high=10),
            self.unimodal_aeeg_normal(),
            self.unimodal_aeeg_suppressed(),
            self.bimodal_aeeg_suppressed(),
            self.bimodal_aeeg_suppressed_with_ictal(),
            self.bimodal_aeeg_burst_suppression(),
            self.bimodal_aeeg_normal(),
            self.low_baseline_suppressed_with_ictal(),
            self.low_baseline_burst_suppression(),
            self.low_baseline_suppressed(),
            self.high_baseline_infrequent_drops_normal(),
            self.high_baseline_frequent_drops_suppressed_with_ictal(),
            self.high_baseline_frequent_drops_burst_suppression()
        ]

    def get_LF_names(self):
        return [
            'bimodal_aeeg', 'low_baseline_aeeg',
            'high_baseline_frequent_drops',
            'very_spiky_aeeg_suppressed_with_ictal',
            'well_separated_aeeg_modes_suppressed_with_ictal',
            'aeeg_NOT_near_zero_normal', 'high_aeeg_baseline_normal_4',
            'high_aeeg_baseline_normal_10', 'unimodal_aeeg_normal',
            'unimodal_aeeg_suppressed', 'bimodal_aeeg_suppressed',
            'bimodal_aeeg_suppressed_with_ictal',
            'bimodal_aeeg_burst_suppression', 'bimodal_aeeg_normal',
            'low_baseline_suppressed_with_ictal',
            'low_baseline_burst_suppression', 'low_baseline_suppressed',
            'high_baseline_infrequent_drops_normal',
            'high_baseline_frequent_drops_suppressed_with_ictal',
            'high_baseline_frequent_drops_burst_suppression'
        ]
