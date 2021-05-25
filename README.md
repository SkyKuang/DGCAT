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
pip install tqdm
```

#### Results
```
| Datasets     | Clear | FGSM   | PGD-20    |      Model                                                   |
| ------------ | -------------- | --------  | ------------------------------------------------------------ |
| CIFAR-10     | 76.4% | 93.47% |  93.47%   | [Modellink](https://drive.google.com)                        |
| CIFAR-100    | 55.6% | 93.47% |  93.54%   | [Modellink](https://drive.google.com)                        |
```
### Train
```
sh ./train.sh
```
### Evaluate
```
sh ./eval.sh
```
