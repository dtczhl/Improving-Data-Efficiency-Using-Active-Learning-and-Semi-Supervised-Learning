import numpy as np
import lightgbm as lgb
from scipy.stats import entropy


def query_index(model, train_set, unqueried_index_set, query_strategy):
    if query_strategy.lower() == "random":
        return np.random.choice(tuple(unqueried_index_set))
    elif query_strategy.lower() == "classification_uncertainty":
        unqueried_index_list = list(unqueried_index_set)
        prob = model.predict(train_set.iloc[unqueried_index_list])
        uncertainty = 1 - prob.max(axis=1)
        sample_index = unqueried_index_list[np.argmax(uncertainty)]
        return sample_index
    elif query_strategy.lower() == "classification_margin":
        unqueried_index_list = list(unqueried_index_set)
        prob = model.predict(train_set.iloc[unqueried_index_list])
        part = np.partition(-prob, 1, axis=1)
        margin = -part[:, 0] + part[:, 1]
        sample_index = unqueried_index_list[np.argmin(margin)]
        return sample_index
    elif query_strategy.lower() == "classification_entropy":
        unqueried_index_list = list(unqueried_index_set)
        prob = model.predict(train_set.iloc[unqueried_index_list])
        sample_index = unqueried_index_list[np.argmax(entropy(prob.T))]
        return sample_index
    else:
        print("Error: unknown strategy", query_strategy)
        exit(-1)