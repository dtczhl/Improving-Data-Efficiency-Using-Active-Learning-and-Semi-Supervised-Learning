% compare

clear, clc

line_spec = {':', 'o-', '.-', '-.', '^-', 'x:', 's-.', 'd-', '^:', 'v-.'};

% al = Active Learning

al_strategies = struct( ...
    'random', 'Random', ...
    'minimize_expected_error', 'Minimize Error', ...
    'density_leastConfident_cosine_2', 'Least Confident, Cosine, \beta:2', ...
    'density_leastConfident_euclidean_1', 'Least Confident, Euclidean, \beta:1', ...
    'density_leastConfident_euclidean_2', 'Least Confident, Euclidean, \beta:2');


data_file_prefix = '../Python/Result/';
data_file_suffix = '.csv';


fields = fieldnames(al_strategies);

my_legend = {};

figure(1), clf, hold on
set(gcf, 'position', [500, 500, 1000, 700])
for k = 1:numel(fields)
    
    filename = fullfile(data_file_prefix, strcat(fields{k}, data_file_suffix));
    data = readmatrix(filename);
    
    
    data = mean(data, 1);
    
    my_legend = [my_legend, al_strategies.(fields{k})];
    

    plot([11:90], data(11:end)*100, line_spec{k}, 'linewidth', 3)
   
    
end

legend(my_legend, 'location', 'southeast', 'fontsize', 22, 'location', 'southeast')
set(gca, 'fontsize', 32, 'ygrid', 'on', 'xgrid', 'on')
xlim([30, 90])
ylim([20, 100])
xlabel('Number of labeled samples')
ylabel('Accuracy (%)')
xticks(0:10:90)
yticks(0:10:100)
% yticks(20:10:100)
hold off
saveas(gcf, './Image/density.png')
