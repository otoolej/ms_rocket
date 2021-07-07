"""

Multiscale ROCKET. See [1] for ROCKET and [2] for modifications. 

Many functions edited from https://github.com/angus924/rocket

[1] A Dempster, F Petitmean, GI Webb. ROCKET: exceptionally fast and accurate time series
classification using random convolutional kernels. Data Mining and Knowledge
Discovery. 2020 Sep;34(5):1454-95.

[2] C Lundy & JM O'Toole (2021) 'Random Convolution Kernels with Multi-Scale Decomposition 
for Preterm EEG Inter-burst Detection' European Signal Poces Conf (EUSIPCO), 2021.



John M. O'Toole, University College Cork 
Started: 25-05-2021 
last update: Time-stamp: <2021-07-06 18:09:42 (otoolej)>
"""
import numpy as np
from numba import njit, prange


@njit("Tuple((f8[:], i4[:], f8[:], i4[:], u1[:]))(i8, i8)")
def generate_kernels(l_signal, n_kernels):
    """
    generate all kernels (from ROCKET code)

    Parameters
    ----------
    l_signal: scalar (int32)
        length of signal
    n_kernels: scalar (int32)
        number of kernels

    Returns
    -------
    kernels : tuple (5 x ndarray) (float64, int32, float64, int32, unit8)
        (weights, l_kerns, bias, scale, high_freq)
    """

    # kernel length is either 7, 9, or 11:
    candidate_lengths = np.array((7, 9, 11), dtype=np.int32)
    l_kerns = np.random.choice(candidate_lengths, n_kernels)

    weights = np.zeros(l_kerns.sum(), dtype=np.float64)
    bias = np.zeros(n_kernels, dtype=np.float64)
    scale = np.zeros(n_kernels, dtype=np.int32)
    high_freq = np.zeros(n_kernels, dtype=np.uint8)

    # iterate over all kernels:
    istart = 0
    for n in range(n_kernels):

        # kernel length:
        n_kern = l_kerns[n]
        iend = istart + n_kern

        # 1. kernel weights (either WGN or low-pass filtered WGN)
        w = np.random.normal(0, 1, n_kern)

        # zero mean the weights:
        weights[istart:iend] = w - w.mean()

        # 2. bias from uniform distribution:
        bias[n] = np.random.uniform(-1, 1)

        # 3. scale:
        scale[n] = np.int32(
            2 ** np.random.uniform(0, np.log2(1 + (l_signal // n_kern)))
        )

        # 4. after filtering, using the high-frequency part (1) or the low frequency part (0):
        high_freq[n] = 1 if np.random.randint(2) == 1 else 0

        istart = iend

    return weights, l_kerns, bias, scale, high_freq


@njit(fastmath=True)
def ma_scaling_lowhigh_freq(x, L):
    """
    filter using moving-average window (with zero-padding)

    Parameters
    ----------
    x: ndarray
        input signal
    L: scalar
        scale value, equal to length of moving average window

    Returns
    -------
    y : ndarray
        filtered signal (trend or detrend component)
    """
    N = len(x)
    Lh = L // 2

    y = np.zeros(N)

    nn = 0
    for n in range(Lh, N + Lh):
        for m in range(L):
            n_m = n - m
            if n_m > -1 and n_m < N:
                # as moving average window w[m] = 1
                y[nn] += x[n_m]

        y[nn] /= L
        nn += 1

    return y



@njit(fastmath=True)
def conv_get_max_ppv(x, w, bias):
    """ 
    convolve x[n] with w[n] (with zero-padding) and return 2 features 

    Parameters
    ----------
    x: ndarray
        input signal
    w: ndarray
        kernel (1D)
    bias: scalar
        bias value

    Returns
    -------
    ppv : scalar
        proportion of positive values (PPV) from x[n] * w[n] (*=convolution)
    c_max : scalar
        maximum value from x[n] * w[n]
    """
    N = len(x)
    L = len(w)
    Lh = L // 2

    c_max = np.NINF
    c_ppv = 0

    for n in range(Lh, (N + Lh)):

        x_tmp = bias
        for m in range(L):
            n_m = n - m
            if n_m > -1 and n_m < N:
                x_tmp = x_tmp + w[m] * x[n_m]

        # only need the max value and percentage of positive values
        # not the total:
        if x_tmp > c_max:
            c_max = x_tmp
        if x_tmp > 0:
            c_ppv += 1

    return c_ppv / N, c_max




@njit()   
def do_ma_filtering_all(X, all_scales, n_epoch, n_scales):
    """ 
    do moving-average filtering for all epochs and store as a matrix

    Parameters
    ----------
    X: ndarray
        matrix of epochs
    all_scales: ndarray
        different values of scale (used as moving-average window size)
    n_epoch: scalar
        number of epochs
    n_scales: scalar
        number of different scale values

    Return
    ------
    X_scales: ndarray
        matrix of epochs after moving-averaging filtering
    """
    X_scales = np.zeros((n_epoch, n_scales), dtype=np.float64)

    for p in range(n_scales):
        if all_scales[p] == 1:
            X_scales[:, p] = X
        else:
            X_scales[:, p] = ma_scaling_lowhigh_freq(X, all_scales[p])

    return X_scales


@njit()
def get_index(x, item):
    """ return the index of the first occurence of item in x """
    for idx, val in enumerate(x):
        if val == item:
            return idx
    return -1
        


@njit(
    "f8[:,:](f8[:,:], Tuple((f8[::1], i4[:], f8[:], i4[:], u1[:])), b1, b1, b1)",
    parallel=True,
    fastmath=True
)
def testing_apply_kernels(X, kernels, do_scale=True, do_high_low_freq=True, do_dilation=False):
    """
    convolve the kernels with the signal segments

    testing the different configurations, as per reference [1] (see header at the top)

    testing_apply_kernels(X, kernels, do_scale=True, do_high_low_freq=True, do_dilation=False)

    is equal to:

    apply_kernels(X, kernels)

    MUST include all 5 input parameters when calling the function (i.e. no optional input parameters)


    Parameters
    ----------
    X: ndarray (M x N) (float64)
        M segments of the time-domain signal
    kernels : tuple (5 x ndarray) (float64, int32, float64, int32, unit8)
        random kernels: (weights, l_kerns, bias, scale, high_freq)
    do_scale : bool (byte)
         include scale, i.e. do the MA filtering
    do_high_low_freq : bool (byte)
         include the high frequency component, in addition to the low-frequency one
    do_dilation : bool (byte)
         include dilation, which downsamples the low-frequency component after MA filtering

    Returns
    -------
    Y : ndarray (M x 2*len(l_kerns))
        2 features from the convolution for each segment in X
    """
    weights, l_kerns, bias, scale, high_freq = kernels

    n_segs, n_epoch = X.shape
    n_kernels = len(l_kerns)

    # what scales are involved?
    all_scales = np.unique(scale)
    n_scales = len(all_scales)

    # 2 features per kernel:
    Y = np.zeros((n_segs, n_kernels * 2), dtype=np.float64)

    for n in prange(n_segs):

        # for weights and features:
        istart1 = 0
        istart2 = 0

        # do the filtering (MA) here and store:
        if do_scale:
            X_scales = do_ma_filtering_all(X[n], all_scales, n_epoch, n_scales)

        
        for m in range(n_kernels):

            if do_scale:
                ip = get_index(all_scales, scale[m])
                if high_freq[m] == 1 and scale[m] > 1 and do_high_low_freq:
                    x_epoch = X[n] - X_scales[:, ip]
                else:
                    if do_dilation:
                        x_epoch = X_scales[::scale[m], ip]
                    else:
                        x_epoch = X_scales[:, ip]

            else:
                x_epoch = X[n]

            iend1 = istart1 + l_kerns[m]
            iend2 = istart2 + 2
            Y[n, istart2:iend2] = conv_get_max_ppv(x_epoch, weights[istart1:iend1], bias[m])


            istart1 = iend1
            istart2 = iend2

    return Y




@njit(
    "f8[:,:](f8[:,:], Tuple((f8[::1], i4[:], f8[:], i4[:], u1[:])))",
    parallel=True,
    fastmath=True
)
def apply_kernels(X, kernels):
    """
    convolve the kernels with the signal segments

    Parameters
    ----------
    X: ndarray (M x N) (float64)
        M segments of the time-domain signal
    kernels : tuple (5 x ndarray) (float64, int32, float64, int32, unit8)
        random kernels: (weights, l_kerns, bias, scale, high_freq)

    Returns
    -------
    Y : ndarray (M x 2*len(l_kerns))
        2 features from the convolution for each segment in X
    """
    weights, l_kerns, bias, scale, high_freq = kernels

    n_segs, n_epoch = X.shape
    n_kernels = len(l_kerns)

    # what scales are involved?
    all_scales = np.unique(scale)
    n_scales = len(all_scales)

    # 2 features per kernel:
    Y = np.zeros((n_segs, n_kernels * 2), dtype=np.float64)

    for n in prange(n_segs):

        # for weights and features:
        istart1 = 0
        istart2 = 0

        # do the filtering (MA) here and store:
        X_scales = do_ma_filtering_all(X[n], all_scales, n_epoch, n_scales)

        
        for m in range(n_kernels):

            ip = get_index(all_scales, scale[m])
            if high_freq[m] == 1 and scale[m] > 1:
                x_epoch = X[n] - X_scales[:, ip]
            else:
                x_epoch = X_scales[:, ip]


            iend1 = istart1 + l_kerns[m]
            iend2 = istart2 + 2
            Y[n, istart2:iend2] = conv_get_max_ppv(x_epoch, weights[istart1:iend1], bias[m])


            istart1 = iend1
            istart2 = iend2

    return Y


