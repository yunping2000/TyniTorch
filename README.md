# TyniTorch
An end-to-end tyni (tiny) neural network implementation with customized CUDA cores, C++ and Python libraries.

## Environment Setup
* nvcc version: 12.5
* gcc version: 13.2.0
* NO dependency on Torch.
```
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -e .
```

## Unit test
```
pytest
```