# SemiSupervisedRegression

This repository contains the code needed to perform a comparison of two semi-supervised regression models: 
- Tri-Training Regressor (to be published)
- COREG [1]

[1] Zhou, Zhi-Hua, and Ming Li. "Semi-supervised regression with co-training." In IJCAI, vol. 5, pp. 908-913. 2005.

The source code of both methods is available in [sslearn](https://github.com/jlgarridol/sslearn).

This comparison corresponds to the experimentation described in the PhD thesis called "Semi-supervised Learning and Survival Analysis in Biological and Health Sciences" by Alicia Olivares Gil. 



## Requirements
# Data
All datasets are available in the `data` folder. 

# Conda environments
all the Python code was executed using the conda environments available in this repository: 
- `uci.yml`: Used to execute all the experiments except the Hierarchical Bayesian tests.
- `baycomp.yml`: Used to execute the Bayesian tests. 

To install these conda environments: 

```
conda env create -f uci.yml
```
To activate the environment: 
```
conda activate uci
```

## Usage
In orer to reproduce the results, follow these steps: 

### 1. Preprocess data
`Preprocess.ipynb` jupyter notebook contains the code to preprocess all datasets. 

### 2. Comparison
Run comparison of semi-supervised regressor methods (and supervised regressor methods as baseline): 

```
python Comparison.py
```
```
python Comparison_Supervised.py
```
### 3. Results
Plot generation is available in `Results.ipynb` jupyter notebook. 

### 4. Hierarchical Bayesian statistical tests
To check whether the differences between the semi-supervised methods and their supervised baseline are signigicant, a Hierarchical Bayesian test is run. 
```
python Statistical_Tests.py
``` 
