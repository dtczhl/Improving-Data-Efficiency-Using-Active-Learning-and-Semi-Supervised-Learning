% show query strategy

clear, clc

query_strategy = 'uncertainty';

x = 11:90;

filename = strcat(query_strategy, '_result_pred.csv');
file_path = fullfile('../Save/', filename);

data = readmatrix(file_path);

mean_val = mean(data);
mean_val = mean_val(x(1):x(end));

figure(2),
plot(x, mean_val, 'linewidth', 2)
xlim([10, 90])
xticks([10:10:90])
yticks([0:0.2:1])
xlabel('Number of samples for training')
ylabel('Accuracy')
set(gca, 'fontsize', 20)
grid on