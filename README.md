# Data Efficiency Project

Collaboration work with Prof. Nitin Nitin.

## Goal

Reducing the number of **labeled** samples, while keeping accuracy of the AI model, using active learning and semi-supervised learning.

## Dataset

1.  Plasma. Given spectrum data (an array of 1868 numbers), predict the plasma dosage (4 classes). Dosage distribution: 27, 27, 30, 30.

2.  EEM. Given spectrum data (an array of 744 numbers), predict the type of solution (2 classes). SP, SE, SEP each has 24 samples.

## ML Model

1.  Plasma. LightGBM.

2.  EEM. LinearSVM

## Experimental Setup

*   Optuna is used to determine the hyper-parameters for each size of samples

*   Initial model is trained using 30 random samples

*   5-fold cross-validation

## Steps

### Plasma

1.  Use optuna to get hyper-parameters for data size from [30, 90]. Each size's hyper-parameter is obtained by 30 trials. Run `Python/Hyper_Parameter/hyper_parameter_init.py`. Results are saved to `Python/Hyper_Parameter/params.pkl`

2.  Run active learning algorithms. See Section Active Learning below.

### EEM

## Random Sampling

We use random sampling as the baseline.

1.  Plasma. Run `Python/active_learning.py random`. Reults are saved to `Result/random.csv`

2. EEM.

## Active Learning

### Least Confident

1.  Plasma. Run `Python/active_learning.py uncertainty_leastConfident`. Results are saved to `Result/uncertainty_leastConfident.csv`.

2.  EEM.

### Entropy

1.  Plasma. Run `Python/active_learning.py uncertainty_entropy`. Results are saved to `Result/uncertainty_entropy.csv`.


### Minimizing Expected Prediction Error

1.  Plasma. Run `Python/active_learning.py minimize_leastConfident`. Results are saved to `Result/minimize_leastConfident.csv`.

### Minimizing Expected Log-loss Error

1.  Plasma. Run `Python/active_learning.py minimize_entropy`. Results are saved to `Result/minimize_entropy.csv`.

## Semi-supervised Learning

### Self Training

1.  Plasma. Run `Python/self_train.py [random|confident|entropy]`. Results are saved to `Result/selfTrain_[random|confident|entropy].csv`.

### Label Spreading

1.  Plasma. Run `Python/label_spreading.py [knn|rbf]`. Results are saved to `Result/selfSpread_[knn|rbf].csv`.


## Combine Active Learning and Semi-supervised Learning

1.  Plasma. Run `Python/combine.py uncertainty_entropy labelSpread_rbf`. Results are saved to `Result/uncertainty_entropy_labelSpread_rbf.csv`.


## Result Analysis and Visualization

Under `Matlab` directory.
