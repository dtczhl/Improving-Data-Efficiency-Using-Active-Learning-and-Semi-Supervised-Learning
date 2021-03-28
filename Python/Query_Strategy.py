import numpy as np
import lightgbm as lgb
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity



def calculate_utility(prob, strategy):
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


def query_index(model, train_set, queried_index_set, unqueried_index_set, query_strategy, target):

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
    # elif query_strategy.lower() == "minimize_expected_error":
    #     unqueried_index_list = list(unqueried_index_set)
    #
    #     # model = objective.best_booster
    #     prob = model.predict(train_set.iloc[unqueried_index_list])
    #     expected_error_arr = np.zeros(len(unqueried_index_list))
    #     i_index = 0
    #     for k, v in zip(unqueried_index_list, prob):
    #         objective_1, objective_2, objective_3, objective_4 \
    #             = deepcopy(objective), deepcopy(objective), deepcopy(objective), deepcopy(objective)
    #         target_1, target_2, target_3, target_4 \
    #             = np.copy(target), np.copy(target), np.copy(target), np.copy(target)
    #         target_1[k], target_2[k], target_3[k], target_4[k] = 0, 1, 2, 3
    #         objective_1.target, objective_2.target, objective_3.target, objective_4.target \
    #             = target_1, target_2, target_3, target_4
    #
    #         queried_index_set_temp = deepcopy(queried_index_set)
    #         queried_index_set_temp.add(k)
    #
    #         objective_1.train_idx, objective_2.train_idx, objective_3.train_idx, objective_4.train_idx = \
    #             list(queried_index_set_temp), list(queried_index_set_temp), \
    #             list(queried_index_set_temp), list(queried_index_set_temp)
    #
    #         study_1 = optuna.create_study(pruner=optuna.pruners.MedianPruner(),
    #                                       sampler=optuna.samplers.RandomSampler(), direction="minimize")
    #         study_1.optimize(objective_1, n_trials=n_trial, callbacks=[objective_1.callback])
    #         model_1 = objective_1.best_booster
    #         prob_1 = model_1.predict(train_set.iloc[unqueried_index_list])
    #         uncertainty_1 = 1 - prob_1.max(axis=1)
    #         uncertainty_1 = np.sum(uncertainty_1)
    #
    #         study_2 = optuna.create_study(pruner=optuna.pruners.MedianPruner(),
    #                                       sampler=optuna.samplers.RandomSampler(), direction="minimize")
    #         study_2.optimize(objective_2, n_trials=n_trial, callbacks=[objective_2.callback])
    #         model_2 = objective_2.best_booster
    #         prob_2 = model_2.predict(train_set.iloc[unqueried_index_list])
    #         uncertainty_2 = 1 - prob_2.max(axis=1)
    #         uncertainty_2 = np.sum(uncertainty_2)
    #
    #         study_3 = optuna.create_study(pruner=optuna.pruners.MedianPruner(),
    #                                       sampler=optuna.samplers.RandomSampler(), direction="minimize")
    #         study_3.optimize(objective_3, n_trials=n_trial, callbacks=[objective_3.callback])
    #         model_3 = objective_3.best_booster
    #         prob_3 = model_3.predict(train_set.iloc[unqueried_index_list])
    #         uncertainty_3 = 1 - prob_3.max(axis=1)
    #         uncertainty_3 = np.sum(uncertainty_3)
    #
    #         study_4 = optuna.create_study(pruner=optuna.pruners.MedianPruner(),
    #                                       sampler=optuna.samplers.RandomSampler(), direction="minimize")
    #         study_4.optimize(objective_4, n_trials=n_trial, callbacks=[objective_4.callback])
    #         model_4 = objective_4.best_booster
    #         prob_4 = model_4.predict(train_set.iloc[unqueried_index_list])
    #         uncertainty_4 = 1 - prob_4.max(axis=1)
    #         uncertainty_4 = np.sum(uncertainty_4)
    #
    #         expected_error = v[0] * uncertainty_1 + v[1] * uncertainty_2 + v[2] * uncertainty_3 + v[3] * uncertainty_4
    #         expected_error_arr[i_index] = expected_error
    #         i_index = i_index + 1
    #
    #     sample_index = np.argmin(expected_error_arr)
    #     sample_index = unqueried_index_list[sample_index]
    #     return sample_index

    else:
        print("Error: unknown strategy", query_strategy)
        exit(-1)
