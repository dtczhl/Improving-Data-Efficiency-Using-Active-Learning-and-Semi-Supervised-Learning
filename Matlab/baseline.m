% baseline

clear, clc

x = 10:5:90;
file_path = '../Save/baseline_result_pred.csv';

data = readmatrix(file_path);

% mean_val = mean(data');
mean_val = data'


figure(1), clf, hold on
plot(x, mean_val, '*-', 'linewidth', 2)
plot(x, 0.25*ones(1, length(x)), 'linewidth', 2)
xlabel('Number of samples for training')
ylabel('Classification accuracy')
xlim([x(1), x(end)])
xticks(x)
yticks([0.2:0.1:1])
legend('LightGBM', 'Random Guess', 'Location', 'northwest')
set(gca, 'fontsize', 20)
grid on
hold off
saveas(gcf, './Image/baseline.png')