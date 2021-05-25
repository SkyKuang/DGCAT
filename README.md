# Towards Robust Adversarial Training via Dual-label Geometry Dispersion

## Introduction
This is the implementation of the
paper ["Towards Robust Adversarial Training via Dual-label Geometry Dispersion"]

## Usage
### Installation
The training environment (PyTorch and dependencies) can be installed as follows:
```
cd DGDAT-master
python setup.py install
```

#### Results
| Datasets     | Clear | FGSM   | PGD-100    | Model                                                                 |
| ------------ | ------|--------| -------- | ----------------------------------------------------------              |
| CIFAR-10     | 90.4% | 77.5%  |  65.6%   | [Modellink](https://pan.baidu.com/s/1YY_86kmFSTZaGK2DHD5FFw):4e8i     |
| CIFAR-100    | 72.8% | 69.3%  |  39.4%   | [Modellink](https://pan.baidu.com/s/1-dfhxl-nL4GnLk7L-U7HVg):olr6     | 
| SVHN         | 95.6% | 91.4%  |  75.1%   | [Modellink](https://pan.baidu.com/s/1-dfhxl-nL4GnLk7L-U7HVg):olr6     |
### Train
```
sh ./train.sh
```
### Evaluate
```
sh ./eval.sh
```
