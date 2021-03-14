% compare

clear, clc

line_spec = {'.-', 'o-', ':', '-.', '+-', 'x:', 's-.', 'd-', '^:', 'v-.'};

% al = Active Learning

al_strategies = struct( ...
    'batch_random_unfair', 'Baseline', ...
    'density_leastConfident_cosine_1', 'Least Confident, \beta:1', ...
    'density_leastConfident_cosine_2', 'Least Confident, \beta:2', ...
    'density_entropy_cosine_1', 'Entropy, \beta:1', ...
    'density_entropy_cosine_2', 'Entropy, \beta:2');


data_file_prefix = '../Save/1/';
data_file_suffix = '_result.csv';


fields = fieldnames(al_strategies);

my_legend = {};

figure(1), clf, hold on
set(gcf, 'position', [500, 500, 1000, 700])
for k = 1:numel(fields)
    
    filename = fullfile(data_file_prefix, strcat(fields{k}, data_file_suffix));
    data = readmatrix(filename);
    
    if strcmp(fields{k}, 'batch_random_unfair') == 1
        data = data';
    end
    
    data = mean(data);
    
    my_legend = [my_legend, al_strategies.(fields{k})];
    
    if strcmp(fields{k}, 'batch_random_unfair') == 1
        plot([11:90], data*100, line_spec{k}, 'linewidth', 3)
    else
        plot([11:90], data(11:end)*100, line_spec{k}, 'linewidth', 3)
    end
    
end

legend(my_legend, 'location', 'southeast', 'fontsize', 22, 'location', 'southeast')
set(gca, 'fontsize', 32, 'ygrid', 'on', 'xgrid', 'on')
xlim([0, 90])
ylim([20, 100])
xlabel('Number of labeled samples')
ylabel('Accuracy (%)')
xticks(0:10:90)
yticks(0:10:100)
% yticks(20:10:100)
hold off
saveas(gcf, './Image/density.png')
