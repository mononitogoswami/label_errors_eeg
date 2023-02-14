import numpy as np
from sklearn import linear_model
from scipy.stats import linregress
from sklearn.neighbors import KDTree
from yaml import load, dump
from yaml import CLoader as Loader, CDumper as Dumper
from pytz import timezone

EASTERN = timezone('US/Eastern')

def find_closest_expert_annotation(expert_labels, patient_id, timestamp):
    """
    A single patient might have multiple expert annotations. 
    Given a timestamp and the ID of a patient, find the most 
    relevant (and closest) expert annotation.
    """
    candidates = expert_labels.loc[expert_labels['id']==patient_id] # Candidates are all expert labels given to a particular patient (specified by patient ID)
    
    if len(candidates) == 0: # If no candidate expert labels
        return np.nan, np.nan, np.nan
    
    ts = []
    for r in candidates.index:
        candidate_dt = EASTERN.localize(candidates.loc[r, 'Timestamp'])    
        ts.append((candidate_dt - timestamp).total_seconds()) # Time differences of the candidates and this window's timestamp
        
    annotation = candidates.loc[candidates.index[np.argmin(ts)], 'Expert annotations']
    min_duration = np.min(ts) # Time elapsed from the last time stamp
    closest_annotation_duration = np.min(np.abs(ts)) # Time duration to the closest timestamp
    
    if min_duration > 3600: # If the expert label is not until one hour after the start of this 
        annotation = None
    
    return annotation, min_duration, closest_annotation_duration 

def fit_robust_line(T):
    # T is a time series
    """
    Fit the RANSAC robust linear regressor 
    Returns: 
    Coefficient, intercept, coefficient of determination (R^2) and predicted line
    """

    ransac = linear_model.RANSACRegressor(base_estimator=linear_model.Ridge(
        alpha=1000))
    # ransac = linear_model.RANSACRegressor()
    ransac.fit(np.arange(len(T)).reshape((-1, 1)), T)
    y = ransac.predict(np.arange(len(T)).reshape((-1, 1)))

    return ransac.estimator_.coef_, ransac.estimator_.intercept_, ransac.estimator_.score(
        y, T), y


# Distribution of Higuchi Fractal Dimension / sample entropy
def _embed(x, order=3, delay=1):
    """Time-delay embedding.
    Parameters
    ----------
    x : 1d-array
        Time series, of shape (n_times)
    order : int
        Embedding dimension (order).
    delay : int
        Delay.
    Returns
    -------
    embedded : ndarray
        Embedded time-series, of shape (n_times - (order - 1) * delay, order)
    """
    N = len(x)
    if order * delay > N:
        raise ValueError("Error: order * delay should be lower than x.size")
    if delay < 1:
        raise ValueError("Delay has to be at least 1.")
    if order < 2:
        raise ValueError("Order has to be at least 2.")
    Y = np.zeros((order, N - (order - 1) * delay))
    for i in range(order):
        Y[i] = x[i * delay:i * delay + Y.shape[1]]
    return Y.T


def _higuchi_fd(x, kmax):
    """Utility function for `higuchi_fd`.
    """
    n_times = x.size
    lk = np.empty(kmax)
    x_reg = np.empty(kmax)
    y_reg = np.empty(kmax)
    for k in range(1, kmax + 1):
        lm = np.empty((k, ))
        for m in range(k):
            ll = 0
            n_max = floor((n_times - m - 1) / k)
            n_max = int(n_max)
            for j in range(1, n_max):
                ll += abs(x[m + j * k] - x[m + (j - 1) * k])
            ll /= k
            ll *= (n_times - 1) / (k * n_max)
            lm[m] = ll
        # Mean of lm
        m_lm = 0
        for m in range(k):
            m_lm += lm[m]
        m_lm /= k
        lk[k - 1] = m_lm
        x_reg[k - 1] = log(1. / k)
        y_reg[k - 1] = log(m_lm)
    higuchi, _, _, _, _ = linregress(x_reg, y_reg)
    return higuchi


def higuchi_fd(x, kmax=6):
    """Higuchi Fractal Dimension.

    Parameters
    ----------
    x : list or np.array
        One dimensional time series.
    kmax : int
        Maximum delay/offset (in number of samples).

    Returns
    -------
    hfd : float
        Higuchi fractal dimension.

    Notes
    -----
    Original code from the `mne-features <https://mne.tools/mne-features/>`_
    package by Jean-Baptiste Schiratti and Alexandre Gramfort.

    This function uses Numba to speed up the computation.

    References
    ----------
    Higuchi, Tomoyuki. "Approach to an irregular time series on the
    basis of the fractal theory." Physica D: Nonlinear Phenomena 31.2
    (1988): 277-283.

    Examples
    --------
    >>> import numpy as np
    >>> from entropy import higuchi_fd
    >>> np.random.seed(123)
    >>> x = np.random.rand(100)
    >>> print(higuchi_fd(x))
    2.0511793572134467
    """
    x = np.asarray(x, dtype=np.float64)
    kmax = int(kmax)
    return _higuchi_fd(x, kmax)


def _app_samp_entropy(x, order, metric='chebyshev', approximate=True):
    """Utility function for `app_entropy`` and `sample_entropy`.
    """
    _all_metrics = KDTree.valid_metrics
    if metric not in _all_metrics:
        raise ValueError('The given metric (%s) is not valid. The valid '
                         'metric names are: %s' % (metric, _all_metrics))
    phi = np.zeros(2)
    r = 0.2 * np.std(x, ddof=0)

    # compute phi(order, r)
    _emb_data1 = _embed(x, order, 1)
    if approximate:
        emb_data1 = _emb_data1
    else:
        emb_data1 = _emb_data1[:-1]
    count1 = KDTree(emb_data1, metric=metric).query_radius(
        emb_data1, r, count_only=True).astype(np.float64)
    # compute phi(order + 1, r)
    emb_data2 = _embed(x, order + 1, 1)
    count2 = KDTree(emb_data2, metric=metric).query_radius(
        emb_data2, r, count_only=True).astype(np.float64)
    if approximate:
        phi[0] = np.mean(np.log(count1 / emb_data1.shape[0]))
        phi[1] = np.mean(np.log(count2 / emb_data2.shape[0]))
    else:
        phi[0] = np.mean((count1 - 1) / (emb_data1.shape[0] - 1))
        phi[1] = np.mean((count2 - 1) / (emb_data2.shape[0] - 1))
    return phi


def sample_entropy(x, order=2, metric='chebyshev'):
    """Sample Entropy.
    Parameters
    ----------
    x : list or np.array
        One-dimensional time series of shape (n_times).
    order : int
        Embedding dimension. Default is 2.
    metric : str
        Name of the distance metric function used with
        :py:class:`sklearn.neighbors.KDTree`. Default is to use the
        `Chebyshev <https://en.wikipedia.org/wiki/Chebyshev_distance>`_
        distance.
    Returns
    -------
    se : float
        Sample Entropy.
    Notes
    -----
    Sample entropy is a modification of approximate entropy, used for assessing
    the complexity of physiological time-series signals. It has two advantages
    over approximate entropy: data length independence and a relatively
    trouble-free implementation. Large values indicate high complexity whereas
    smaller values characterize more self-similar and regular signals.
    The sample entropy of a signal :math:`x` is defined as:
    .. math:: H(x, m, r) = -\\log\\frac{C(m + 1, r)}{C(m, r)}
    where :math:`m` is the embedding dimension (= order), :math:`r` is
    the radius of the neighbourhood (default = :math:`0.2 * \\text{std}(x)`),
    :math:`C(m + 1, r)` is the number of embedded vectors of length
    :math:`m + 1` having a
    `Chebyshev distance <https://en.wikipedia.org/wiki/Chebyshev_distance>`_
    inferior to :math:`r` and :math:`C(m, r)` is the number of embedded
    vectors of length :math:`m` having a Chebyshev distance inferior to
    :math:`r`.
    Note that if ``metric == 'chebyshev'`` and ``len(x) < 5000`` points,
    then the sample entropy is computed using a fast custom Numba script.
    For other distance metric or longer time-series, the sample entropy is
    computed using a code from the
    `mne-features <https://mne.tools/mne-features/>`_ package by Jean-Baptiste
    Schiratti and Alexandre Gramfort (requires sklearn).
    References
    ----------
    Richman, J. S. et al. (2000). Physiological time-series analysis
    using approximate entropy and sample entropy. American Journal of
    Physiology-Heart and Circulatory Physiology, 278(6), H2039-H2049.
    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html
    Examples
    --------
    Sample entropy with order 2.
    >>> from entropy import sample_entropy
    >>> import numpy as np
    >>> np.random.seed(1234567)
    >>> x = np.random.rand(3000)
    >>> print(sample_entropy(x, order=2))
    2.192416747827227
    Sample entropy with order 3 using the Euclidean distance.
    >>> from entropy import sample_entropy
    >>> import numpy as np
    >>> np.random.seed(1234567)
    >>> x = np.random.rand(3000)
    >>> print(sample_entropy(x, order=3, metric='euclidean'))
    2.724354910127154
    """
    x = np.asarray(x, dtype=np.float64)
    phi = _app_samp_entropy(x, order=order, metric=metric, approximate=False)
    return -np.log(np.divide(phi[1], phi[0]))


class Config:
    def __init__(
            self,
            config_file_path='../config.yaml'):
        """Class to read and parse the config.yml file
		"""
        self.config_file_path = config_file_path

    def parse(self):
        with open(self.config_file_path, 'rb') as f:
            self.config = load(f, Loader=Loader)
        return self.config

    def save_config(self):
        with open(self.config_file_path, 'w') as f:
            dump(self.config, f)