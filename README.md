## Overview
This package provides implementations for various generative models.

## Installation
This package has been tested with Python 3.11 and can be installed using `pip`:

```bash
user@account:~/generative-models$ python3 -m venv generative
user@account:~/generative-models$ source generative/bin/activate
(generative) user@account:~/instadeep-test$ pip install -r requirements.txt
```

It can also be installed using `conda`:

```bash
(base) C:\User\generative-models> conda create -f environment.yml
(base) C:\User\generative-models> conda activate generative
(generative) C:\User\generative-models> pip install -r requirements.txt
```

In some cases (due to a bug in the TF 2.16 pip installation), Conda may not set up the `LD_LIBRARY_PATH` environment variable properly, leading to Tensorflow library errors on import. In this case, `LD_LIBRARY_PATH` should be set manually:

```shell
(generative) user@account:~/generative-models$ conda env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cublas/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_cupti/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_nvcc/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cufft/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/curand/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cusolver/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cusparse/lib
```

The package can then be installed locally:
```bash
(protein) user@account:~/generative-models$ pip install -e .
```
