import numpy as np
import lightgbm as lgb


def query_index(model, train_set, unqueried_index_set, query_strategy):
    if query_strategy.lower() == "random":
        return np.random.choice(tuple(unqueried_index_set))
    elif query_strategy.lower() == "uncertainty":
        unqueried_index_list = list(unqueried_index_set)
        prob = model.predict(train_set.iloc[unqueried_index_list])
        uncertainty = 1 - prob.max(axis=1)
        sample_index = unqueried_index_list[np.argmax(uncertainty)]
        return sample_index
    else:
        print("Error: unknown strategy", query_strategy)
        exit(-1)