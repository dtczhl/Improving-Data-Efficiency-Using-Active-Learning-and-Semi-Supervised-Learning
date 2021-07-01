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

*   Initial model is trained using 20 random samples

*   5-fold cross-validation

## Steps

### Plasma

1.  Use optuna to get hyper-parameters. `Python/Hyper_Parameter/hyper_parameter_init.py`. Results are saved to `Python/Hyper_Parameter/params.pkl` [Error]

2.  Run active learning algorithms. See Section Active Learning below.

### EEM


## Active Learning

### Uncertainty based

- Least Confident
- Margin
- Entropy

### Information Density


- Informativenss: [Least Confident | Margin | Entropy]
- Representativeness: [Cosine Similarity | Euclidean Distance | Pearson Similarity]
- Importance Ratio <img src="https://bit.ly/2SA9n8Y" align="center" border="0" alt="\beta" width="17" height="19" />:   [0.5, 1, 2]




### Minimizing Expected Error

## Semi-supervised Learning


## Combine Active Learning and Semi-supervised Learning
