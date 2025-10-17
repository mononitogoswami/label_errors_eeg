####################################################
#  Code to featurize time-series data
####################################################
"""
Advanced time-series feature extraction for EEG signal analysis.

This module implements sophisticated signal processing techniques to extract
meaningful features from EEG time-series data including baseline trends,
spectral characteristics, complexity measures, and statistical properties.
"""

from typing import List, Tuple, Union, Optional
import numpy as np
from scipy.stats.mstats import winsorize
from sklearn.mixture import GaussianMixture
from scipy.signal import find_peaks
from .utils import fit_robust_line, sample_entropy, higuchi_fd


class TimeSeriesFeaturizer(object):
    """
    Advanced feature extraction for EEG time-series signals.

    Extracts comprehensive features from EEG signals including baseline trends,
    spectral properties, complexity measures, statistical characteristics,
    and peak analysis. Uses robust signal processing methods to handle
    artifacts and missing data.

    Features extracted include:
    - Baseline characteristics (slope, intercept, average/final values)
    - FFT coefficients for spectral analysis
    - Statistical measures (percentiles, IQR, standard deviation)
    - Complexity measures (Higuchi fractal dimension, sample entropy)
    - Signal band analysis (fraction below amplitude thresholds)
    - Peak detection and characterization
    - Gaussian mixture model parameters

    Attributes:
        timeseries (np.ndarray): Processed time-series data
        n (int): Length of time-series
        baseline_exists (bool): Whether baseline fitting succeeded
        slope (float): Baseline slope from robust linear regression
        y_intercept (float): Baseline y-intercept
        peaks (np.ndarray): Indices of detected signal peaks
        prominences (np.ndarray): Peak prominence values

    Example:
        >>> signal = np.random.randn(1000) + 5
        >>> featurizer = TimeSeriesFeaturizer(signal)
        >>> features = featurizer.featurize()
        >>> feature_names = featurizer.get_feature_names()
        >>> print(f"Extracted {len(features)} features")
    """

    def __init__(self, timeseries: np.ndarray, remove_artifacts: bool = True) -> None:
        """
        Initialize featurizer with time-series data.

        Args:
            timeseries: Input EEG time-series as numpy array
            remove_artifacts: Whether to apply winsorization to remove outliers/artifacts

        Raises:
            ValueError: If timeseries is empty or invalid
        """
        # Input validation
        if not isinstance(timeseries, (np.ndarray, list)) or len(timeseries) == 0:
            raise ValueError("Timeseries must be non-empty numpy array or list")

        if len(timeseries) < 50:
            raise ValueError(f"Timeseries too short ({len(timeseries)} samples). Minimum 50 samples required.")

        if remove_artifacts:
            self.timeseries = winsorize(a=timeseries, limits=[0.01, 0.01],
                               inplace=False)  # To remove artifacts (1% trimming from each end)
            # winsorize returns a masked array, compressed() methods returns only valid entries
        else:
            self.timeseries = timeseries

        self.n = len(self.timeseries)
        self.baseline_exists = True

        try:
            self.slope, self.y_intercept, _, self.y = fit_robust_line(self.timeseries)  # Baseline
            self.peaks, _ = find_peaks(self.timeseries.reshape((-1,)), prominence=2, width=1)  # Peaks
            self.prominences = self.timeseries[self.peaks].reshape(
                (-1, )) - self.y.reshape((-1, ))[self.peaks]
        except:  # Baseline cannot be computed
            self.baseline_exists = False

    def get_baseline_features(self) -> Tuple[float, float, float, float]:
        """
        Extract baseline trend characteristics from EEG signal.

        Uses robust linear regression to fit a baseline trend and extract
        slope, intercept, and amplitude characteristics.

        Returns:
            Tuple[float, float, float, float]: (slope, y_intercept, average_baseline, final_baseline)
                - slope: Rate of change in baseline amplitude (mV/sample)
                - y_intercept: Initial baseline amplitude (mV)
                - average_baseline: Mean baseline amplitude across signal (mV)
                - final_baseline: Final baseline amplitude (mV)
        """
        if not self.baseline_exists:
            return np.nan, np.nan, np.nan, np.nan

        final_baseline_value = float(self.y[-1])
        average_baseline_value = np.sum(self.y) / len(self.y)
        return float(self.slope), float(
            self.y_intercept), average_baseline_value, final_baseline_value

    def get_fft_features(self):
        fft_coefs = np.fft.fft(self.timeseries).real[1:4].squeeze()
        if len(fft_coefs) < 3:
            return np.nan, np.nan, np.nan
        return fft_coefs

    def get_dispersion_and_central_tendency_features(self):
        per_5, median, per_95 = np.percentile(a=self.timeseries, q=[5, 50, 95])
        iqr = per_95 - per_5
        stdev = np.std(self.timeseries)
        stderror = stdev / np.sqrt(self.n)
        return per_5, median, per_95, iqr, stdev, stderror

    def area_sum_features(self):
        return np.trapz(self.timeseries.compressed()), np.sqrt(np.mean(
            self.timeseries**2)).squeeze()

    def complexity_features(self):
        return higuchi_fd(self.timeseries.squeeze(),
                          kmax=6), sample_entropy(self.timeseries.squeeze(), order=2)

    def signal_band_features(self):
        return np.sum(self.timeseries < 1) / self.n, np.sum(
            self.timeseries < 3) / self.n, np.sum(self.timeseries < 5) / self.n, np.sum(
                self.timeseries < 10) / self.n

    def peak_features(self):
        if not self.baseline_exists:
            return np.nan, np.nan, np.nan
        if len(self.peaks) > 0:  # There may be no peaks at all
            n_peaks_vha = len(self.peaks[self.prominences >= 15])
            n_peaks_ha = len(self.peaks[self.prominences >= 10])
            n_peaks_la = len(self.peaks[np.logical_and(self.prominences > 5,
                                                       self.prominences < 10)])
        else:
            n_peaks_vha, n_peaks_ha, n_peaks_la = 0, 0, 0
        return n_peaks_vha, n_peaks_ha, n_peaks_la

    def get_gaussian_mixture_features(self):
        obj = GaussianMixture(n_components=2)
        obj.fit(self.timeseries.reshape(-1, 1))
        gaussian_weight_1, gaussian_weight_2 = obj.weights_.squeeze()
        gaussian_mean_1, gaussian_mean_2 = obj.means_.squeeze()
        return gaussian_weight_1, gaussian_weight_2, gaussian_mean_1, gaussian_mean_2

    def featurize(self):
        """
        Compute all features
        """
        baseline_slope, baseline_y_intercept, average_baseline_value, final_baseline_value = self.get_baseline_features()
        fft_coef_1, fft_coef_2, fft_coef_3 = self.get_fft_features()
        per_5, median, per_95, iqr, stdev, stderror = self.get_dispersion_and_central_tendency_features()
        auc, rms_sum = self.area_sum_features()
        hfd, se = self.complexity_features()
        frac_below_1mV, frac_below_3mV, frac_below_5mV, frac_below_10mV = self.signal_band_features()
        n_peaks_vha, n_peaks_ha, n_peaks_la = self.peak_features()
        gaussian_weight_1, gaussian_weight_2, gaussian_mean_1, gaussian_mean_2 = self.get_gaussian_mixture_features()

        return [
            baseline_slope, baseline_y_intercept, average_baseline_value,
            final_baseline_value, fft_coef_1, fft_coef_2, fft_coef_3, per_5,
            median, per_95, iqr, stdev, stderror, auc, rms_sum, hfd, se,
            n_peaks_vha, n_peaks_ha, n_peaks_la, frac_below_1mV,
            frac_below_3mV, frac_below_5mV, frac_below_10mV, gaussian_weight_1,
            gaussian_weight_2, gaussian_mean_1, gaussian_mean_2
        ]

    def get_feature_names(self):
        return [
            'Slope of baseline', 'Y-intercept of baseline',
            'Average value of baseline', 'Final value of baseline',
            'First fft coefficient', 'Second fft coefficient',
            'Third fft coefficient', '5 percentile', 'Median', '95 percentile',
            'Inter-quartile range', 'Standard deviation', 'Standard error',
            'Area under curve', 'Root mean square sum', 'Higuchi FD',
            'Sample entropy', 'Number of very high amplitude peaks',
            'Number of high amplitude peaks', 'Number of low amplitude peaks',
            'Fraction below 1mV', 'Fraction below 3mV', 'Fraction below 5mV',
            'Fraction below 10mV', 'Gaussian weight 1', 'Gaussian weight 2',
            'Gaussian mean 1', 'Gaussian mean 2'
        ]
