% compare

clear, clc

line_spec = {':', 'o-', '.-', '-.', '^-'};

% al = Active Learning

al_strategies = struct( ...
    'random', 'Random', ...
    'uncertainty_leastConfident', 'Least Confident', ...
    'uncertainty_entropy', 'Entropy');

data_file_prefix = '../Python/Result/';
data_file_suffix = '.csv';


fields = fieldnames(al_strategies);

my_legend = {};
all_data = [];


for k = 1:numel(fields)
    
    filename = fullfile(data_file_prefix, strcat(fields{k}, data_file_suffix));
    data = readmatrix(filename);

    mean_data = mean(data);
    
    all_data = [all_data; mean_data];
    my_legend = [my_legend, al_strategies.(fields{k})];
     
end

figure(1), clf, hold on
set(gcf, 'position', [500, 500, 1000, 650])

x = 11:90;

for i_row = 1:size(all_data, 1)
    data = all_data(i_row, 11:end);
    plot(x, data *100, line_spec{i_row}, 'linewidth', 2)
end

h_legend = columnlegend(2, my_legend, 'location', 'northwest', 'fontsize', 22);
h_legend.Position = [0.42, 0.14, 0.6, 0.2];

set(gca, 'fontsize', 32, 'ygrid', 'on', 'xgrid', 'on')
xlim([30, 90])
ylim([20, 100])
xlabel('Number of human-annotated samples')
ylabel('Accuracy (%)')
xticks(10:10:90)
yticks(20:10:100)
title('Uncertainty Sampling')

axes('Position', [0.6, 0.45, 0.25, 0.25])
box on, hold on

for i_row = 1:size(all_data, 1)
    data = all_data(i_row, 11:end);
    indexOfInterest = (x >= 60) & (x <= 80);
    plot(x(indexOfInterest), 100 * data(indexOfInterest), line_spec{i_row}, 'linewidth', 2)
end
set(gca, 'fontsize', 18)
xticks([60:5:80])
ylim([84, 90])
yticks([84:2:90])

hold off
saveas(gcf, './Image/uncertainty.png')
