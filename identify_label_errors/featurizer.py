####################################################
#  Code to featurize time-series data
####################################################

from math import log, floor
import numpy as np
from scipy.stats.mstats import winsorize
from sklearn.mixture import GaussianMixture
from scipy.signal import find_peaks
from utils import fit_robust_line, sample_entropy, higuchi_fd


class Featurizer(object):
    def __init__(self, T: np.ndarray, remove_artifacts: bool = True):
        if remove_artifacts:
            self.T = winsorize(a=T, limits=[0.01, 0.01],
                               inplace=False)  # To remove artifacts
            # winsorize returns a masked array, compressed() methods returns only valid entries
        else:
            self.T = T

        self.n = len(self.T)
        self.baseline_exists = True

        try:
            self.slope, self.y_intercept, _, self.y = fit_robust_line(
                self.T)  # Baseline
            self.peaks, _ = find_peaks(self.T.reshape((-1, )),
                                       prominence=2,
                                       width=1)  # Peaks
            self.prominences = self.T[self.peaks].reshape(
                (-1, )) - self.y.reshape((-1, ))[self.peaks]
        except:  # Baseline cannot be computed
            self.baseline_exists = False

    def get_baseline_features(self):
        if not self.baseline_exists:
            return np.nan, np.nan, np.nan, np.nan

        final_baseline_value = float(self.y[-1])
        average_baseline_value = np.sum(self.y) / len(self.y)
        return float(self.slope), float(
            self.y_intercept), average_baseline_value, final_baseline_value

    def get_fft_features(self):
        fft_coefs = np.fft.fft(self.T).real[1:4].squeeze()
        if len(fft_coefs) < 3:
            return np.nan, np.nan, np.nan
        return fft_coefs

    def get_dispersion_and_central_tendency_features(self):
        per_5, median, per_95 = np.percentile(a=self.T, q=[5, 50, 95])
        iqr = per_95 - per_5
        stdev = np.std(self.T)
        stderror = stdev / np.sqrt(self.n)
        return per_5, median, per_95, iqr, stdev, stderror

    def area_sum_features(self):
        return np.trapz(self.T.compressed()), np.sqrt(np.mean(
            self.T**2)).squeeze()

    def complexity_features(self):
        return higuchi_fd(self.T.squeeze(),
                          kmax=6), sample_entropy(self.T.squeeze(), order=2)

    def signal_band_features(self):
        return np.sum(self.T < 1) / self.n, np.sum(
            self.T < 3) / self.n, np.sum(self.T < 5) / self.n, np.sum(
                self.T < 10) / self.n

    def peak_features(self):
        if not self.baselineExists:
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
        obj.fit(self.T.reshape(-1, 1))
        gaussian_weight_1, gaussian_weight_2 = obj.weights_.squeeze()
        gaussian_mean_1, gaussian_mean_2 = obj.means_.squeeze()
        return gaussian_weight_1, gaussian_weight_2, gaussian_mean_1, gaussian_mean_2

    def featurize(self):
        """
        Compute all features
        """
        baseline_slope, baseline_y_intercept, average_baseline_value, final_baseline_value = self.baseline_features(
        )
        fft_coef_1, fft_coef_2, fft_coef_3 = self.FFT_features()
        per_5, median, per_95, iqr, stdev, stderror = self.dispersion_and_central_tendency_features(
        )
        auc, rms_sum = self.area_sum_features()
        hfd, se = self.complexity_features()
        frac_below_1mV, frac_below_3mV, frac_below_5mV, frac_below_10mV = self.signal_band_features(
        )
        n_peaks_vha, n_peaks_ha, n_peaks_la = self.peak_features()
        gaussian_weight_1, gaussian_weight_2, gaussian_mean_1, gaussian_mean_2 = self.gaussian_mixture_features(
        )

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
            'First FFT coefficient', 'Second FFT coefficient',
            'Third FFT coefficient', '5 percentile', 'Median', '95 percentile',
            'Inter-quartile range', 'Standard deviation', 'Standard error',
            'Area under curve', 'Root mean square sum', 'Higuchi FD',
            'Sample entropy', 'Number of very high amplitude peaks',
            'Number of high amplitude peaks', 'Number of low amplitude peaks',
            'Fraction below 1mV', 'Fraction below 3mV', 'Fraction below 5mV',
            'Fraction below 10mV', 'Gaussian weight 1', 'Gaussian weight 2',
            'Gaussian mean 1', 'Gaussian mean 2'
        ]
