% compare

clear, clc

% al = Active Learning

al_strategies = struct( ...
    'batch_random', 'Batch Random', ...
    'random', 'Random', ...
    'uncertainty_leastConfident', 'Uncertainty: Least Confident', ...
    'uncertainty_margin', 'Uncertainty: Margin', ...
    'uncertainty_entropy', 'Uncertainty: Entropy', ...
    'density_leastConfident_cosine_05', 'Density: Least Confident + Cosine + \beta: 0.5', ...
    'density_leastConfident_cosine_1', 'Density: Least Confident + Cosine + \beta: 1', ...
    'density_leastConfident_cosine_2', 'Density: Least Confident + Cosine + \beta 2', ...
    'density_margin_cosine_05', 'Density: Margin + Cosine + \beta: 0.5', ...
    'density_margin_cosine_1', 'Density: Margin + Cosine + \beta: 1', ...
    'density_margin_cosine_2', 'Density: Margin + Cosine + \beta: 2', ...
    'density_entropy_cosine_05', 'Density: Entropy + Cosine + \beta: 0.5', ...
    'density_entropy_cosine_1', 'Density: Entropy + Cosine + \beta: 1', ...
    'density_entropy_cosine_2', 'Density: Entropy  + Cosine + \beta: 2');

data_file_prefix = '../Save/1/';
data_file_suffix = '_result.csv';


fields = fieldnames(al_strategies);

my_legend = {};

figure(1), clf, hold on
for k = 1:numel(fields)
    
    if strcmp(fields{k}, 'batch_random') == 1
        continue;
    end
    
    filename = fullfile(data_file_prefix, strcat(fields{k}, data_file_suffix));
    data = readmatrix(filename);
    data = mean(data);
    
    my_legend = [my_legend, al_strategies.(fields{k})];
    
    plot([11:90], data(11:end)*100, 'linewidth', 3)
    
end
legend(my_legend, 'location', 'southeast')
set(gca, 'fontsize', 24, 'ygrid', 'on', 'xgrid', 'on')
xlabel('Number of samples for training')
ylabel('Accuracy (%)')
xticks(10:10:90)
yticks(20:10:100)
hold off


return

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