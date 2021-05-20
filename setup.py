from setuptools import setup, find_packages

setup(
    name='dual-label-geometry-dispersion-adversarial-training',
    version='0.0.1',
    install_requires=[
        'torch',
        'torchvision',
        'scipy',
        'tqdm',
    ],
    packages=find_packages(),
)
