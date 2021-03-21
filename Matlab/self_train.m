% Self Train

clear, clc

line_spec = {'.-', 'o-', ':', '-.'};

% al = Active Learning

al_strategies = struct( ...
    'batch_random_unfair', 'Baseline', ...
    'selfTrain_leastConfident_temp', 'Self Train: Least Confident', ...
    'selfTrain_entropy_temp', 'Self Train: Entropy');

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
        data = mean(data);
    end
    
%     data = mean(data);
    
    my_legend = [my_legend, al_strategies.(fields{k})];
    
    if strcmp(fields{k}, 'batch_random_unfair') == 1
        plot([11:90], data*100, line_spec{k}, 'linewidth', 3)
    else
        plot([11:90], data(11:end)*100, line_spec{k}, 'linewidth', 3)
    end
    
end

legend(my_legend, 'location', 'southeast', 'fontsize', 22)
set(gca, 'fontsize', 32, 'ygrid', 'on', 'xgrid', 'on')
xlim([0, 90])
ylim([20, 100])
xlabel('Number of labeled samples')
ylabel('Accuracy (%)')
xticks(10:10:90)
yticks(20:10:100)
hold off
saveas(gcf, './Image/self_train.png')
