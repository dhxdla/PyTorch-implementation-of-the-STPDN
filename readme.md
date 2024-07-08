# STPDN: Spatio-Temporal Pattern Decomposition Network with Fluctuation Awareness for Robust Traffic Flow Forecasting

## Introduction
This repository contains the PyTorch implementation of the paper "STPDN: Spatio-Temporal Pattern Decomposition Network with Fluctuation Awareness for Robust Traffic Flow Forecasting," presented at ECAI 2024.

## Requirements
To install the required packages, run:
```bash
pip install -r requirements.txt
```

## Datasets
Download the dataset from [Google Drive](https://drive.google.com/drive/folders/1LqJcEkPjQqnha4gbg5Ql40speW7mDjJ9?usp=drive_link) and place it in the `./data` directory.

For detailed code to generate noise data, refer to `./data/gen_noise.ipynb`.

## Model Training and Evaluation
To run the experiment, use the following command:
```bash
python main.py -file [config_file]
```
For example:
```bash
python main.py -file pems04
```

All configuration parameters are located in the `./config` folder. Name the config file after the dataset. To load a trained model directly, set the `mode` parameter in the config file to `test`.

## Save Model
The training model is saved at each step in the `best_model` folder. The best model is named `./best_model/model.pt`.

## Citation
If you use this code, please cite our paper:
```
[Stay tuned.]
```

