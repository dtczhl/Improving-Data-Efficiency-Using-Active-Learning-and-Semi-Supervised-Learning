% compare

clear, clc

line_spec = {'.-', 'o-', ':', '-.', '^-'};

% al = Active Learning

al_strategies = struct( ...
    'random', 'Random', ...
    'uncertainty_leastConfident', 'Least Confident');

data_file_prefix = '../Python/Result/';
data_file_suffix = '.csv';


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

    mean_data = mean(data);
    
    my_legend = [my_legend, al_strategies.(fields{k})];
    
    if strcmp(fields{k}, 'batch_random_unfair') == 1
        plot([11:90], mean_data*100, line_spec{k}, 'linewidth', 3)
    else
        plot([11:90], mean_data(11:end)*100, line_spec{k}, 'linewidth', 3)
%         errorbar([11:90], mean_data(11:end)*100, std_data(11:end)*100, line_spec{k}, 'linewidth', 3)
    end
    
end

legend(my_legend, 'location', 'southeast', 'fontsize', 22)
set(gca, 'fontsize', 32, 'ygrid', 'on', 'xgrid', 'on')
xlim([0, 90])
ylim([20, 100])
xlabel('Number of human labeled samples')
ylabel('Accuracy (%)')
xticks(10:10:90)
yticks(20:10:100)
hold off
saveas(gcf, './Image/uncertainty.png')
