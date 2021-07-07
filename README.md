# msROCKET: random convolutional kernels with multi-scale decomposition

Python code which extends the ROCKET method to include multi-scale decomposition. See
following reference for more details:

`C Lundy & JM O'Toole (2021) 'Random Convolution Kernels with Multi-Scale Decomposition for
Preterm EEG Inter-burst Detection' European Signal Poces Conf (EUSIPCO), 2021.`

Please cite the above reference if using this code to generate new results.

```bibtex
@inproceedings{Lundy2021,
  author = {Lundy, Christopher and O'Toole, John M.},
  title = {Random Convolution Kernels with Multi-Scale Decomposition for Preterm {EEG} Inter-burst Detection},
  booktitle = {European Signal Processing Conference (EMBC)},
  month = {aug},
  pages = {1--5},
  publisher = {IEEE},
  year = {2021},
}
```

Requires Python 3 with NumPy and Numba packages. 

The original ROCKET method is available on [github](https://github.com/angus924/rocket)
with details in [1](#references).

---
[Requirements](#requires) | [Examples](#examples) | [Licence](#licence) |
[References](#references) | [Contact](#contact)


## Requires
Developed and tested with Python 3.9 and:
+ NumPy (version 1.19.5)
+ Numba (version 0.53.1)

May work with older versions but not tested.


## Install
Options to install:
+ download the repository and copy the module `ms_rocket/ms_kernel_fns.py` to where its needed.
+ or, download and install an editable version with `pip`:

within this directory, do:
```python
pip3 install -e .
```

+ or install directly from github

```python
pip3 install git+https://github.com/otoolej/ms_rocket
```



## Examples
Generate random kernels and convolve with the input signal

```python
import numpy as np
import ms_kernel_fns as ms

# 1. generate random test signal: 800 segments of length-128
X = np.random.randn(800, 128)

# 2. generate 10,000 kernels (default number) for segments of length 128 samples:
kerns = ms.generate_kernels(128, 10000)

# 3. convolve the 10,000 kernels with each 800 segment
X_train = ms.apply_kernels(X, kerns)
```

The matrix `X_train` is the feature matrix of size `800 x 20,000`, as 2 features for each
kernel. This feature set is then combined using a linear classifier, e.g. ridge regression.


### Train and testing a classifier

The following is an example of using msROCKET for a detection problem. Here we detect a
frequency-modulated signal in white Gaussian noise.

Requires `sklearn.linear_model.RidgeRegression` and `sklearn.model_selection.train_test_split`.


Taken from the file `ms_rocket/example_msrocket.py`:
```python
 """ example of msROCKET for detection of time-varying sinusoidal components in noise """
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier

from example_msrocket import gen_data
import ms_kernel_fns as ms


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
```


## Licence
```
Copyright (c) 2021, John M. O'Toole, University College Cork
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

  Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.

  Neither the name of the University College Cork nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```



## References
1. A Dempster, F Petitjean & GI Webb (2020). ROCKET: exceptionally fast and accurate time
   series classification using random convolutional kernels. Data Mining and Knowledge
   Discovery, 34(5),
   1454-1495. [DOI:10.1007/s10618-020-00701-z](https://doi.org/10.1007/s10618-020-00701-z)
  



## Contact

John M. O'Toole

Neonatal Brain Research Group,  
[INFANT Research Centre](https://www.infantcentre.ie/),  
Department of Paediatrics and Child Health,  
Room 2.19 UCC Academic Paediatric Unit, Cork University Hospital,  
University College Cork,  
Ireland

- email: jotoole AT ucc _dot ie 


