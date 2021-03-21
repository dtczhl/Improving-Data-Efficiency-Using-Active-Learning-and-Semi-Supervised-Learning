# Data Description







Run = 5

* `Query By Disagreement`. `query_by_disagreement_*`
  * `query_by_disagreement_result_{threshold}.csv`: overall accuracy. 2-D matrix: [run, 90]. The first 10 columns are 0.
  * `query_by_disagreement_keep_accuracy_{threshold}.csv`: validation accuracy versus the number of samples. 2-D matrix: [run, 90*5]. The first 10 columns of every 90 columns are 0.
  * `query_by_disagreement_keep_size_{threshold}.csv`: number of accepted samples for training. 2-D matrix: [run, 90*5]. The first 10 columns of every 90 columns are 0.



Folders

* 1
    ```python
    n_run = 5
    n_trial = 5
    ```
