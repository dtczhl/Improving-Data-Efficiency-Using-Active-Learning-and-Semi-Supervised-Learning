# Data Efficiency Project

Data and source code for our paper:

[**Spectroscopy Approaches for Food Safety Applications: Improving Data Efficiency Using Active Learning and Semi-Supervised Learning**](https://www.frontiersin.org/articles/10.3389/frai.2022.863261/abstract), published in *frontiers in Artificial Intelligence - AI in Food, Agriculture and Water*, 2022

Authors: 
Huanle Zhang, Nicharee Wisuthiphaet, Hemiao Cui, Nitin Nitin, Xin Liu, and Qing Zhao


## Goal

Reducing the number of **labeled** samples, while keeping accuracy of the AI model, using active learning and semi-supervised learning.

## Dataset

1. Plasma. Given spectrum data (an array of 1868 numbers), predict the plasma dosage (4 classes). Dosage distribution: 27, 27, 30, 30.

2. EEM. Given spectrum data (an array of 744 numbers), predict the type of solution (2 classes). SP, SE, SEP each has 24 samples.

3. EEM Multi-class. Given spectrum data (an array of 620 numbers), predict four classes. Dosage classification: 40, 40, 40, 40

## ML Model

1. Plasma. LightGBM.

2. EEM. LinearSVM

3. EEM multi-class. Logistic Regression classifier

## Experimental Setup

*   Optuna is used to determine the hyper-parameters for each size of samples

*   Initial model is trained using 30 random samples

*   5-fold cross-validation

## Steps

### Plasma

1.  Use optuna to get hyper-parameters for data size from [30, 90]. Each size's hyper-parameter is obtained by 30 trials. Run `Python/Hyper_Parameter/hyper_parameter_init.py`. Results are saved to `Python/Hyper_Parameter/params.pkl`

2.  See Section Active Learning, Semi-supervised Learning, Combine Active Learning and Semi-supervised Learning.

### EEM

1.  Use optuna to get hyper-parameters for data size from [20, 57]. Each size's hyper-parameter is obtained by 30 trials. Run `Python/Hyper_Parameter/hyper_parameter_init_eem.py`. Results are saved to `Python/Hyper_Parameter/params_eem.pkl`

2.  See Section Active Learning, Semi-supervised Learning, Combine Active Learning and Semi-supervised Learning.

### EEM Multi

1.  Use optuna to get hyper-parameters for data size from [15, 128]. Each size's hyper-parameter is obtained by 30 trials. Run `Python/Hyper_Parameter/hyper_parameter_init_eem_multi.py`. Results are saved to `Python/Hyper_Parameter/params_eem_multi.pkl`

2.  See Section Active Learning, Semi-supervised Learning, Combine Active Learning and Semi-supervised Learning.


## Random Sampling

We use random sampling as the baseline.

1. Plasma. Run `Python/active_learning.py random`. Results are saved to `Result/random.csv`

2. EEM.

3. EEM multi-class. Run `Python/active_learning_eem_multi.py random`. Results are saved to `Result/random_eem_multi.csv`



## Active Learning

### Least Confident

1. Plasma. Run `Python/active_learning.py uncertainty_leastConfident`. Results are saved to `Result/uncertainty_leastConfident.csv`.

2. EEM.

3. EEM multi. Run `Python/active_learning_eem_multi.py uncertainty_leastConfident`. Results are saved to `Result/uncertainty_leastConfident_eem_multi.csv`.

### Entropy

1. Plasma. Run `Python/active_learning.py uncertainty_entropy`. Results are saved to `Result/uncertainty_entropy.csv`.

2. EEM.

3. EEM multi. Run `Python/active_learning_eem_multi.py uncertainty_entropy`. Results are saved to `Result/uncertainty_entropy_eem_multi.csv`.

### Minimizing Expected Prediction Error

1. Plasma. Run `Python/active_learning.py minimize_leastConfident`. Results are saved to `Result/minimize_leastConfident.csv`.

2. EEM

3. EEM multi. Run `Python/active_learning_eem_multi.py minimize_leastConfident`. Results are saved to `Result/minimize_leastConfident_eem_multi.csv`.

### Minimizing Expected Log-loss Error

1. Plasma. Run `Python/active_learning.py minimize_entropy`. Results are saved to `Result/minimize_entropy.csv`.

2. EEM

3. EEM multi. Run `Python/active_learning_eem_multi.py minimize_entropy`. Results are saved to `Result/minimize_entropy_eem_multi.csv`.

## Semi-supervised Learning

### Self Training

1. Plasma. Run `Python/self_train.py [random|confident|entropy]`. Results are saved to `Result/selfTrain_[random|confident|entropy].csv`.

2. EEM

3. EEM multi. Run `Python/self_train_eem_multi.py [random|confident|entropy]`. Results are saved to `Result/selfTrain_[random|confident|entropy]_eem_multi.csv`.

### Label Spreading

1. Plasma. Run `Python/label_spreading.py [knn|rbf]`. Results are saved to `Result/selfSpread_[knn|rbf].csv`.

2. EEM

3. EEM multi. Run `Python/label_spreading_eem_multi.py [knn|rbf]`. Results are saved to `Result/selfSpread_[knn|rbf]_eem_multi.csv`.

## Combine Active Learning and Semi-supervised Learning

1. Plasma. Run `Python/combine.py uncertainty_entropy [labelSpread_rbf|selfTrain_random]`. Results are saved to `Result/uncertainty_entropy_[labelSpread_rbf|selfTrain_random].csv`.

2. EEM

3. EEM multi. Run `Python/combine_eem_multi.py uncertainty_entropy labelSpread_rbf`. Results are saved to `Result/uncertainty_entropy_labelSpread_rbf_eem_multi.csv`.

## Result Analysis and Visualization

Under `Matlab` directory.

