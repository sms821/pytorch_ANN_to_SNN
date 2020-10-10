# Pytorch ANN to SNN
A Pytorch-based library for simulation of rate-encoded deep spiking neural networks. This library mostly implements the ANN to SNN conversion method described in [Conversion of Continuous-Valued Deep Networks to Efficient Event-Driven Networks for Image Classification] (https://www.frontiersin.org/articles/10.3389/fnins.2017.00682/full). It also supports a hybrid SNN-ANN simulation in which the initial layers are spiking and the latter layers are non-spiking.

## Citation
If using for an academic publication, please consider citing "NEBULA: A Neuromorphic Spin-Based Ultra-Low Power Architecture for SNNs and ANNs", in International Symposium on Computer Architecture, 2020.
```
@inproceedings{10.1109/ISCA45697.2020.00039,
author = {Singh, Sonali and Sarma, Anup and Jao, Nicholas and Pattnaik, Ashutosh and Lu, Sen and Yang, Kezhou and Sengupta, Abhronil and Narayanan, Vijaykrishnan and Das, Chita R.},
title = {NEBULA: A Neuromorphic Spin-Based Ultra-Low Power Architecture for SNNs and ANNs},
year = {2020},
isbn = {9781728146614},
publisher = {IEEE Press},
url = {https://doi.org/10.1109/ISCA45697.2020.00039},
doi = {10.1109/ISCA45697.2020.00039},
```

## What is this repository for?
- Validating a pre-trained pytorch model (ANN).
- Converting ANN to SNN and simulating it. 
- Plotting graphs showing the correlation between SNN and ANN models.
- Simulating hybrid SNN-ANN models.

## Installation
Python 3.6 is needed along with other packages mentioned in `requirements.txt`. CUDA acceleration is supported but not required to run the code. To install the required packages run the following command:
```
pip install -r requirements.txt
```

## Examples
In order to run the code, enter the following command from the terminal:
```
python main.py --config-file configs/lenet5.ini
```
The `configs` folder contains `.ini` files in which appropriate flags need to be set to achieve the desired task. Location of the-pretrained networks along with simulation parameter details are also specified here. Please refer to `configs/tutorial.ini` for a description of all the flags. Some of the pre-trained models used in this implementation can be found [here] (https://psu.box.com/s/4inwor4mj9900kvps42278hl8zdpnhin).
