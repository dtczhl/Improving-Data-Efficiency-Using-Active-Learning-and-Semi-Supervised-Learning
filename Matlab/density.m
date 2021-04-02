% compare

clear, clc

line_spec = {':', 'o-', '.-', '-.', '^-', 'x:', 's-.', 'd-', '^:', 'v-.'};

% al = Active Learning

al_strategies = struct( ...
    'random', 'Random', ...
    'density_leastConfident_cosine_1', 'Least Confident, \beta:1', ...
    'density_leastConfident_cosine_2', 'Least Confident, \beta:2', ...
    'density_entropy_cosine_1', 'Entropy, \beta:1', ...
    'density_entropy_cosine_2', 'Entropy, \beta:2');


data_file_prefix = '../Python/Result/';
data_file_suffix = '.csv';


fields = fieldnames(al_strategies);

my_legend = {};
all_data = [];

for k = 1:numel(fields)
    
    filename = fullfile(data_file_prefix, strcat(fields{k}, data_file_suffix));
    data = readmatrix(filename);
    
    mean_data = mean(data);
    
    my_legend = [my_legend, al_strategies.(fields{k})];
    all_data = [all_data; mean_data];

end

figure(1), clf, hold on
set(gcf, 'position', [500, 500, 1000, 600])

x = 11:90;

for i_row = 1:size(all_data, 1)
    data = all_data(i_row, 11:end);
    plot(x, data *100, line_spec{i_row}, 'linewidth', 2)
end

h_legend = columnlegend(2, my_legend, 'location', 'northwest', 'fontsize', 22);
h_legend.Position = [0.3, 0.09, 0.72, 0.3];

set(gca, 'fontsize', 32, 'ygrid', 'on', 'xgrid', 'on')
xlim([30, 90])
ylim([20, 100])
xlabel('Number of human labeled samples')
ylabel('Accuracy (%)')
xticks(10:10:90)
yticks(20:10:100)

axes('Position', [0.6, 0.48, 0.25, 0.25])
box on, hold on

for i_row = 1:size(all_data, 1)
    data = all_data(i_row, 11:end);
    indexOfInterest = (x >= 60) & (x <= 80);
    plot(x(indexOfInterest), 100 * data(indexOfInterest), line_spec{i_row}, 'linewidth', 2)
end
set(gca, 'fontsize', 18)
xticks([60:5:80])
% ylim([80, 90])
yticks([84:2:90])

hold off
saveas(gcf, './Image/density.png')
