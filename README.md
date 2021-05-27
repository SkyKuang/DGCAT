# Towards Robust Adversarial Training via Dual-label Geometry Dispersion

## Introduction
This is the implementation of the
paper ["Towards Robust Adversarial Training via Dual-label Geometry Dispersion"]
Abstract: Adversarial training has been demonstrated to be highly effective in defending against adversarial example attacks. However, such robustness comes with a price of a larger generalization gap, making adversarial training less practical in various real-world applications. To this end, existing endeavors mainly treat each training example independently, which ignores the relationship between inter-samples and does not take the defending capability to the full potential. Different from existing works, this paper tackles these problems by making full use of the geometry relationship between inter-samples and transfering the geometric information of inter-samples into the training on adversarial examples. Furthermore, a dual-label method is proposed to leverage true and wrong labels of adversarial example to jointly supervise the adversarial learning process. To further characterize this dual-label supervised learning, theoretical explanation provided for the internal working mechanism of adversarial samples. We have conducted extensive experiments on benchmark datasets, which well demonstrate that the proposed approach significantly alleviates the aforementioned challenges.

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
| SVHN         | 96.8% | 95.9%  |  76.1%   | [Modellink](https://pan.baidu.com/s/1pmjG4ddW5Ic-NERfDjYxzQ):zup6     |
### Train
```
sh ./train.sh
```
### Evaluate
```
sh ./eval.sh
```
