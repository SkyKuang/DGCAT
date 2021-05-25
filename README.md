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
#### CIFAR-10

| Full Model   | Flops &#8595; | Accuracy | Pruned Model                                                 |
| ------------ | ----------------- | -------- | ------------------------------------------------------------ |
| VGG16        | 76.4%             | 93.47%   | [Modellink](https://drive.google.com/drive/folders/1GWR56Aoc08r3eUUwSub1_lxJ0Z06dWyd?usp=sharing) |
| ResNet56     | 55.6%             | 93.54%   | [Modellink](https://drive.google.com/drive/folders/1NSnJnLGWsSJLiVCksk1OnOK2iVGRfLyg?usp=sharing) |
| ResNet110    | 66.0%             | 94.12%   | [Modellink](https://drive.google.com/drive/folders/1h-eSUbtJ_xO3wlnQ7J3Pl8bBsuTEw9LJ?usp=sharing) |
| MobileNet-V2 | 29.2%             | 95.28%   | [Modellink](https://drive.google.com/drive/folders/1Q78kM5U8Tz-nonCLbBisVrke97OGIIai?usp=sharing) |
