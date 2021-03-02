% query by disagreement


clear, clc


data_pred = readmatrix('../Save/query_by_disagreement_result.csv');
data_keep_size = readmatrix('../Save/query_by_disagreement_keep_size.csv');
data_keep_size = reshape(data_keep_size, 90, 5);
data_keep_size = data_keep_size';


figure(1)
plot(data_pred)

figure(2)
plot([1:90],data_keep_size)