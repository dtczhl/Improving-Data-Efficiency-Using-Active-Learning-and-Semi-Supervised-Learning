% compare

clear, clc


batch_random_file_path = '../Save/baseline_result_20.csv';
al_random_file_path = '../Save/random_result_20.csv';
al_uncertainty_file_path = '../Save/classification_margin_result_20.csv';

batch_random_data = readmatrix(batch_random_file_path);
batch_random_data = mean(batch_random_data, 2);

al_random_data = readmatrix(al_random_file_path);
al_random_data = mean(al_random_data);

al_uncertainty_data = readmatrix(al_uncertainty_file_path);
al_uncertainty_data = mean(al_uncertainty_data);

figure(1), clf, hold on
plot([10:5:90], batch_random_data * 100, '*-','linewidth', 2)
plot([11:90], al_random_data(11:end) * 100, 'linewidth', 2)
plot([11:90], al_uncertainty_data(11:end)* 100, 'linewidth', 2)
xlabel('Number of samples for training')
ylabel('Accuracy (%)')
xticks(10:10:90)
yticks(20:10:100)
legend('Batch Random', 'Incremental Random', 'Incremental Uncertainty', 'location', 'southeast')
set(gca, 'fontsize', 18)
grid on
xlim([10, 90])
hold off