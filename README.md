# Plasma Project

Collaboration work with Prof. Nitin Nitin.


## Background

Given spectrum data (an array of length 1868), predict the plasma dosage (4 classes).

We use `LightGBM` model

We adopt 5-fold cross-validation

## Goal

Reducing the number of *labeled* samples, while keeping accuracy of the AI model

## Method

We use active learning and semi-supervised learning

### Active Learning

#### Uncertainty based

- Least Confident
- Margin
- Entropy

#### Information Density


- Informativenss: [Least Confident | Margin | Entropy]
- Representativeness: [Cosine Similarity | Euclidean Distance | Pearson Similarity]
- Importance Ratio <img src="https://bit.ly/2SA9n8Y" align="center" border="0" alt="\beta" width="17" height="19" />:   [0.5, 1, 2]




#### Minimizing Expected Error

### Semi-supervised Learning


### Combine Active Learning and Semi-supervised Learning
