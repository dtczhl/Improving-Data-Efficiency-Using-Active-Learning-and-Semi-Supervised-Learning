import numpy as np
import lightgbm as lgb
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity

# Supported query strategy
# random
# Uncertainty: classification_uncertainty, classification_margin, classification_entropy
# Information Density: information_density_[entropy]_[cosine]_[x]

similar_arr = None


def query_index(model, train_set, unqueried_index_set, query_strategy):

    global similar_arr

    if query_strategy.lower() == "random":
        return np.random.choice(tuple(unqueried_index_set))
    # Uncertainty
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
    # Information Density
    elif query_strategy[0:len("information_density")].lower() == "information_density":

        sub_fields = query_strategy.lower().split("_")
        base_query_method = sub_fields[2]
        similarity_metric = sub_fields[3]
        beta = float(sub_fields[4])

        if similar_arr is None:

            print("!!!!!!!!!! Should only be called once")

            # construct similar array
            similar_arr = np.zeros(len(train_set))

            if similarity_metric.lower() == "cosine":
                for i_sample in range(len(train_set)):
                    temp_sum = 0
                    for j_sample in range(len(train_set)):
                        if i_sample == j_sample:
                            continue
                        sim_value = np.squeeze(cosine_similarity(train_set.iloc[i_sample].to_numpy().reshape(1, -1),                                               train_set.iloc[j_sample].to_numpy().reshape(1, -1)))[()]
                        temp_sum = temp_sum + sim_value
                    similar_arr[i_sample] = temp_sum / (len(train_set) - 1)
            else:
                print("Unknown similarity", similarity_metric)
                exit(-1)

            similar_arr = np.power(similar_arr, beta)

        if base_query_method.lower() == "entropy":

            unqueried_index_list = list(unqueried_index_set)
            prob = model.predict(train_set.iloc[unqueried_index_list])
            entropy_temp = entropy(prob.T)
            information_density = np.multiply(entropy_temp, similar_arr[unqueried_index_list])
            sample_index = unqueried_index_list[np.argmax(information_density)]
            return sample_index
        else:
            print("Unknown Base Query", base_query_method.lower())
            exit(-1)
    else:
        print("Error: unknown strategy", query_strategy)
        exit(-1)
