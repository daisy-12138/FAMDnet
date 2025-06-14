# Frequency Assisted Multi-Scale Dual-Stream Network for Low-quality Deepfake Detection

## Introduction

This is the official pytorch implementation of Frequency Assisted Multi-Scale Dual-Stream Network for Low-quality Deepfake Detection.

## Clone this repository

```bash
git clone https://github.com/daisy-12138/FAMDnet
```

## Install the required dependencies

```bash
pip install -r requirements.txt
```

## Datasets
FF++ and Celeb-DF datasets can be available at https://github.com/SCLBD/DeepfakeBench.
WildDeepfake dataset can be available at https://github.com/OpenTAI/wild-deepfake.

## Training and Evaluation

```
python run.py --cfg FAMDnet.yaml
```
