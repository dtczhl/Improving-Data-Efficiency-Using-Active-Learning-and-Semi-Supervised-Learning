% baseline

clear, clc

x = 10:10:90;
file_path = '../Save/result_pred_2.csv';

data = readmatrix(file_path);

mean_val = mean(data');

figure(1), clf, hold on
plot(x, mean_val, '*-', 'linewidth', 2)
plot(x, 0.25*ones(1, length(x)), 'linewidth', 2)
xlabel('Number of samples for training')
ylabel('Classification accuracy')
xlim([10, 90])
xticks([10:10:90])
set(gca, 'fontsize', 20)
grid on
hold off