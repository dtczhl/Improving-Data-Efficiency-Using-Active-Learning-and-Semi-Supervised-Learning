# Data Efficiency Project

Collaboration work with Prof. Nitin Nitin.

## Goal

Reducing the number of **human-labeled** samples, while keeping accuracy of the AI model, using active learning and semi-supervised learning.

## Dataset

1.  Plasma. Given spectrum data (an array of 1868 numbers), predict the plasma dosage (4 classes).

2.  EEM. Given spectrum data (an array of XXX numbers), predict the type of solution (2 classes).

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


### Minimizing Expected Error

## Semi-supervised Learning


## Combine Active Learning and Semi-supervised Learning

## Result Analysis and Visualization

Under `Matlab` directory.
