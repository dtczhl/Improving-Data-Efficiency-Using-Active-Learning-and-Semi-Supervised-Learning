% compare

clear, clc


batch_random_file_path = '../Save/1/batch_random_result.csv';
random_file_path = '../Save/1/random_result.csv';
uncertainty_least_confident_file_path = '../Save/1/classification_uncertainty_result.csv';
uncertainty_margin_file_path = '../Save/1/classification_margin_result.csv';
uncertainty_entropy_file_path = '../Save/1/classification_entropy_result.csv';
density_entropy_cosine_05 = '../Save/1/information_density_entropy_cosine_0.5_result.csv';
density_entropy_cosine_1 = '../Save/1/information_density_entropy_cosine_1_result.csv';
density_entropy_cosine_2 = '../Save/1/information_density_entropy_cosine_2_result.csv';



batch_random_data = readmatrix(batch_random_file_path);
batch_random_data = mean(batch_random_data, 2);

random_data = readmatrix(random_file_path);
random_data = mean(random_data);

uncertainty_least_confident_data = readmatrix(uncertainty_least_confident_file_path);
uncertainty_least_confident_data = mean(uncertainty_least_confident_data);

uncertainty_margin_data = readmatrix(uncertainty_margin_file_path);
uncertainty_margin_data = mean(uncertainty_margin_data);

uncertainty_entropy_data = readmatrix(uncertainty_entropy_file_path);
uncertainty_entropy_data = mean(uncertainty_entropy_data);

density_entropy_cosine_05_data = readmatrix(density_entropy_cosine_05);
density_entropy_cosine_05_data = mean(density_entropy_cosine_05_data);

density_entropy_cosine_1_data = readmatrix(density_entropy_cosine_1);
density_entropy_cosine_1_data = mean(density_entropy_cosine_1_data);

density_entropy_cosine_2_data = readmatrix(density_entropy_cosine_2);
density_entropy_cosine_2_data = mean(density_entropy_cosine_2_data);


figure(1), clf, hold on
plot([10:90], batch_random_data * 100, '*-','linewidth', 3)
plot([11:90], random_data(11:end) * 100, 'linewidth', 3)
plot([11:90], uncertainty_least_confident_data(11:end)* 100, 'linewidth', 3)
plot([11:90], uncertainty_margin_data(11:end)* 100, 'linewidth', 3)
plot([11:90], uncertainty_entropy_data(11:end)* 100, 'linewidth', 3)
plot([11:90], density_entropy_cosine_05_data(11:end)* 100, 'linewidth', 3)
plot([11:90], density_entropy_cosine_1_data(11:end)* 100, 'linewidth', 3)
plot([11:90], density_entropy_cosine_2_data(11:end)* 100, 'linewidth', 3)
xlabel('Number of samples for training')
ylabel('Accuracy (%)')
xticks(10:10:90)
yticks(20:10:100)
legend('Batch Random', 'Random', ...
    'Uncertainty: Least Confident', 'Uncertainty: Margin', 'Uncertainty: Entropy', ...
    'Density: entropy + cosine + 0.5', 'Density: entropy + cosine + 1', 'Density: entropy + cosine + 2', ...
    'location', 'southeast')
set(gca, 'fontsize', 24)
grid on
xlim([10, 90])
hold off