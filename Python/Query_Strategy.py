import numpy as np
import lightgbm as lgb
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity

import copy

end_n_sample = 90

def calculate_utility(prob, strategy):
    # return [n_row, 1]
    if strategy == "confident":
        pred_confident = prob.max(axis=1)
        return pred_confident
    elif strategy == "margin":
        part = np.partition(-prob, 1, axis=1)
        margin_diff = -part[:, 0] + part[:, 1]
        return margin_diff
    elif strategy == "entropy":
        entropy_info = entropy(prob.T)
        return entropy_info
    else:
        print("-------- Error: Unknown strategy={}".format(strategy))
        exit(-1)


def query_index(model, train_set, queried_index_set, unqueried_index_set, query_strategy, target,
                val_idx=None, hyper_params=None):

    if query_strategy.lower() == "random":
        return np.random.choice(tuple(unqueried_index_set))

    # Uncertainty Sampling
    elif query_strategy.lower().startswith("uncertainty"):
        sub_fields = query_strategy.lower().split("_")
        base_query_method = sub_fields[1]

        unqueried_index_list = list(unqueried_index_set)

        prob = model.predict(train_set.iloc[unqueried_index_list])

        if base_query_method == "leastconfident":
            utility = calculate_utility(prob, "confident")
            sample_index_unordered = np.argmin(utility)
            sample_index = unqueried_index_list[sample_index_unordered]
            return sample_index
        elif base_query_method == "margin":
            utility = calculate_utility(prob, "margin")
            sample_index_unordered = np.argmin(utility)
            sample_index = unqueried_index_list[sample_index_unordered]
            return sample_index
        elif base_query_method == "entropy":
            utility = calculate_utility(prob, "entropy")
            sample_index_unordered = np.argmax(utility)
            sample_index = unqueried_index_list[sample_index_unordered]
            return sample_index
        else:
            print("****** Error: Unknown base_query_method={}".format(base_query_method))
            exit(-1)
    elif query_strategy.lower().startswith("density"):
        sub_fields = query_strategy.lower().split("_")
        base_query_method = sub_fields[1]
        similarity_metric = sub_fields[2]
        beta = float(sub_fields[3])

        unqueried_index_list = list(unqueried_index_set)

        prob = model.predict(train_set.iloc[unqueried_index_list])

        # construct similar array
        similar_arr = np.zeros(len(unqueried_index_list))

        for i_sample in range(len(unqueried_index_list)):
            temp_sum = 0
            for j_sample in range(len(unqueried_index_list)):
                if i_sample == j_sample:
                    continue

                if similarity_metric.lower() == "cosine":
                    sim_value = np.squeeze(cosine_similarity(train_set.iloc[unqueried_index_list[i_sample]].to_numpy().reshape(1, -1),
                                                             train_set.iloc[unqueried_index_list[j_sample]].to_numpy().reshape(1, -1)))[()]
                elif similarity_metric.lower() == "pearson":
                    sim_value = np.corrcoef(train_set.iloc[unqueried_index_list[i_sample]].to_numpy().reshape(1, -1),
                                            train_set.iloc[unqueried_index_list[j_sample]].to_numpy().reshape(1, -1))[0][1]
                elif similarity_metric.lower() == "euclidean":
                    sim_value = np.squeeze(train_set.iloc[unqueried_index_list[i_sample]].to_numpy().reshape(1, -1)
                                           - train_set.iloc[unqueried_index_list[j_sample]].to_numpy().reshape(1, -1))
                    sim_value = np.sum(np.square(sim_value))
                else:
                    print("Unknown similarity", similarity_metric)
                    exit(-1)
                temp_sum = temp_sum + sim_value

            similar_arr[i_sample] = temp_sum / len(unqueried_index_list)

        similar_arr = np.power(similar_arr, beta)

        if base_query_method.lower() == "leastconfident":
            utility = calculate_utility(prob, "confident")
            uncertainty = 1 - utility
            information_density = np.multiply(uncertainty, similar_arr)
            sample_index = unqueried_index_list[np.argmax(information_density)]
            return sample_index
        elif base_query_method.lower() == "margin":
            utility = calculate_utility(prob, "margin")
            information_density = np.multiply(utility, similar_arr)
            sample_index = unqueried_index_list[np.argmax(information_density)]
            return sample_index
        elif base_query_method.lower() == "entropy":
            utility = calculate_utility(prob, "entropy")
            information_density = np.multiply(utility, similar_arr)
            sample_index = unqueried_index_list[np.argmax(information_density)]
            return sample_index
        else:
            print("Unknown Base Query", base_query_method.lower())
            exit(-1)

    # Minimize Expected Error
    elif query_strategy.lower().startswith("minimize"):
        sub_fields = query_strategy.lower().split("_")
        base_query_method = sub_fields[1]

        unqueried_index_list = list(unqueried_index_set)
        prob = model.predict(train_set.iloc[unqueried_index_list])
        expected_error_arr = np.zeros(len(unqueried_index_list))

        i_index = 0

        for k, v in zip(unqueried_index_list, prob):

            target_1, target_2, target_3, target_4 \
                = np.copy(target), np.copy(target), np.copy(target), np.copy(target)
            target_1[k], target_2[k], target_3[k], target_4[k] = 0, 1, 2, 3

            queried_index_set_temp = copy.deepcopy(queried_index_set)
            queried_index_set_temp.add(k)

            # Last number,
            if len(queried_index_set_temp) > len(hyper_params):
                return unqueried_index_list[0]

            train_data_1 = lgb.Dataset(train_set.iloc[list(queried_index_set_temp)],
                                       label=target_1[list(queried_index_set_temp)])
            train_data_2 = lgb.Dataset(train_set.iloc[list(queried_index_set_temp)],
                                       label=target_2[list(queried_index_set_temp)])
            train_data_3 = lgb.Dataset(train_set.iloc[list(queried_index_set_temp)],
                                       label=target_3[list(queried_index_set_temp)])
            train_data_4 = lgb.Dataset(train_set.iloc[list(queried_index_set_temp)],
                                       label=target_4[list(queried_index_set_temp)])

            valid_data = lgb.Dataset(train_set.iloc[val_idx], label=target[val_idx])
            model_1 = lgb.train(hyper_params[len(queried_index_set_temp)], train_data_1, valid_sets=[valid_data],
                                verbose_eval=False)
            valid_data = lgb.Dataset(train_set.iloc[val_idx], label=target[val_idx])
            model_2 = lgb.train(hyper_params[len(queried_index_set_temp)], train_data_2, valid_sets=[valid_data],
                                verbose_eval=False)
            valid_data = lgb.Dataset(train_set.iloc[val_idx], label=target[val_idx])
            model_3 = lgb.train(hyper_params[len(queried_index_set_temp)], train_data_3, valid_sets=[valid_data],
                                verbose_eval=False)
            valid_data = lgb.Dataset(train_set.iloc[val_idx], label=target[val_idx])
            model_4 = lgb.train(hyper_params[len(queried_index_set_temp)], train_data_4, valid_sets=[valid_data],
                                verbose_eval=False)

            prob_1 = model_1.predict(train_set.iloc[unqueried_index_list])
            prob_2 = model_2.predict(train_set.iloc[unqueried_index_list])
            prob_3 = model_3.predict(train_set.iloc[unqueried_index_list])
            prob_4 = model_4.predict(train_set.iloc[unqueried_index_list])

            if base_query_method == "leastconfident":

                utility_1 = calculate_utility(prob_1, "confident")
                error_1 = np.sum(1 - utility_1)

                utility_2 = calculate_utility(prob_2, "confident")
                error_2 = np.sum(1 - utility_2)

                utility_3 = calculate_utility(prob_3, "confident")
                error_3 = np.sum(1 - utility_3)

                utility_4 = calculate_utility(prob_4, "confident")
                error_4 = np.sum(1 - utility_4)

            elif base_query_method == "entropy":

                utility_1 = calculate_utility(prob_1, "entropy")
                error_1 = np.sum(utility_1)

                utility_2 = calculate_utility(prob_2, "entropy")
                error_2 = np.sum(utility_2)

                utility_3 = calculate_utility(prob_3, "entropy")
                error_3 = np.sum(utility_3)

                utility_4 = calculate_utility(prob_4, "entropy")
                error_4 = np.sum(utility_4)

            else:
                print("Unknown base_query_method={}".format(base_query_method))
                exit(-1)

            expected_error = v[0] * error_1 + v[1] * error_2 + v[2] * error_3 + v[3] * error_4
            expected_error_arr[i_index] = expected_error
            i_index = i_index + 1

        sample_index = np.argmin(expected_error_arr)
        sample_index = unqueried_index_list[sample_index]
        return sample_index

    else:
        print("Error: unknown strategy", query_strategy)
        exit(-1)
