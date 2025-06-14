from setuptools import find_packages, setup

setup(
    name='FAMDnet',
    version='0.1.1',
    requires= ['timm',
        'fvcore',
        'albumentations',
        'kornia',
        'simplejson',
        'tensorboard',
    ],
    packages=find_packages(),
    license="apache 2.0")
