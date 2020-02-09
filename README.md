# Bastien Girardet - Bachelor Thesis - Cryptocurrencies volatility forecasting using heterogeneous models based on multi-layer neural network adn classical ARCH family models

# Installation
First we need to install pipenv

```shell
$ brew install pipenv
```

or

```shell
$ pip3 install pipenv 
```

Then provided the Pipenv file in the repository, we can install all the requirements by doing in the directory:

```shell
$ pipenv install
```
# Scripts are to be run in the correct order

## 0-generate-graphs
In this notebook we generate all the graphs for this thesis concerning descriptive stats, log returns distribution, historical volatility distribution, etc..

Figures are saved in the folder */figs*

## 1-bruteforce-ts-model (Heavy computation)
This python script bruteforce all arch-type possible models
The results are saved in */models/ts/fits*

## 2-forecast_variance_with_garch_model_eval
This file recompute the variance of each models to make forecast on the testing sample
Results are saved in */mcs/data/models/TS/*

## 3-compute-mcs-ts
We process the results from script number 2 to find the superior set of classical arch-types models.
Results are saved in *./mcs/results/\*-top-5\*.csv/*

## 4-find_best_model
We get back the results of the models from the metadata saved in script \#1

## 5-forecast_variance_with_ARCH_and_plot
We plot the top models

## 6-forecast_vol_with_ann (Heavy computation)
Here we compute all combination possible with the different hyperpamaters
(e.g. model 5 - Drop out rate 0.4 - Normal - ANN-GARCH(1,1)

## 7-compute-mcs-hetero
Here we compute the mcs for the heterogenous models computed in the previous script.
Results are saved in */mcs/results/crypto-top-5-formatted.csv*

## 8-compute-hetero-and-graph
We plot the top models

## Libraries
Cryptov2

Own library made for the purpose for this thesis can be found in *libaries/Cryptov2.py*
Compute various indicators

