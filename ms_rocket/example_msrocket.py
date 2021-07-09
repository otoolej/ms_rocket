"""
Testing functions for msROCKET. See [1] for details.

[1] C Lundy & JM O'Toole (2021) 'Random Convolution Kernels with Multi-Scale Decomposition 
for Preterm EEG Inter-burst Detection' European Signal Proc Conf (EUSIPCO), 2021.

requires: numpy, sklearn, ms_rocket


John M. O' Toole, University College Cork
Started: 06-07-2021
last update: Time-stamp: <2021-07-09 17:00:01 (otoolej)>
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier

from ms_rocket import ms_kernel_fns as ms



def gen_LFM_signal(N=1024):
    """ generate a linear frequency-modulated signal """

    # start and stop frequencies:
    fstart = 0.1
    fstop = 0.4

    n = np.arange(N) 
    x = np.cos( 2 * np.pi * (fstart * n + ((fstop - fstart) / (2 * N)) * (n ** 2)) )

    return x


def gen_data(N=1024):
    """ generate a mock dataset with a linear frequency-modulated (LFM) signal in noise """

    # 1. generate LFM signal with additive white-Gaussian noise:
    x = gen_LFM_signal(N)
    n = np.random.randn(N) * 1.5
    x += n
    x /= np.std(x)

    # 2. generate WGN
    y = np.random.randn(N)
    y /= np.std(y)

    # 3. split into epochs of length 128 samples with 50% overlap
    l_epoch = 128
    overlap = 50
    l_overlap = np.floor(l_epoch * overlap / 100).astype(int)

    x_epochs = np.lib.stride_tricks.sliding_window_view(x, l_epoch)[::l_overlap]
    y_epochs = np.lib.stride_tricks.sliding_window_view(y, l_epoch)[::l_overlap]

    # 4. define the class labels
    n_epochs = x_epochs.shape[0]
    class_labels = np.zeros(2 * n_epochs, dtype=np.uint8)
    class_labels[:n_epochs] = 1

    # 5. combine the 2 classes
    X = np.vstack((x_epochs, y_epochs))

    return X, class_labels



def msrocket_example_det_LFM():
    """ example of msROCKET for detection of time-varying sinusoidal components in noise """

    # -------------------------------------------------------------------
    #  1. generate the data:
    # -------------------------------------------------------------------
    X, y = gen_data(2 ** 15)

    # -------------------------------------------------------------------
    #  2. split 80/20 for training/testing
    # -------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)    

    # -------------------------------------------------------------------
    #  3. generate kernels:
    # -------------------------------------------------------------------
    kerns = ms.generate_kernels(X_train.shape[1], 10000)

    # -------------------------------------------------------------------
    #  4. convolve the training data with the kernels
    # -------------------------------------------------------------------
    X_train_c = ms.apply_kernels(X_train, kerns)


    # -------------------------------------------------------------------
    #  5. train the ridge regression classifier
    # -------------------------------------------------------------------
    ml_model = RidgeClassifier(normalize=True)        
    ml_model.fit(X_train_c, y_train)

    train_acc = ml_model.score(X_train_c, y_train)
    

    # -------------------------------------------------------------------
    #  6. test
    # -------------------------------------------------------------------
    X_test_c = ms.apply_kernels(X_test, kerns)    
    acc = ml_model.score(X_test_c, y_test)

    print('ACCURACY: training={:.3f} | testing={:.3f}'.format(train_acc, acc))



if __name__ == "__main__":
    msrocket_example_det_LFM()
