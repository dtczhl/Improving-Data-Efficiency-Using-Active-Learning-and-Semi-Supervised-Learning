% compare

clear, clc

my_line_style = {':', '-', '-', '-.', '-'};
my_marker = {'none', 'o', '+', 'none', '^'};
line_spec = {':', 'o-', '+-', '-.', '^-'};


% al = Active Learning

al_strategies = struct( ...
    'random', 'Baseline', ...
    'uncertainty_entropy_labelSpread_rbf', 'Entropy + Graph: RBF');

data_file_prefix = '../Python/Result/';
data_file_suffix = '.csv';


fields = fieldnames(al_strategies);

my_legend = {};
all_data = [];
all_std = [];

figure(1), clf, hold on
set(gcf, 'position', [500, 500, 1000, 650])

x = 11:90;

cMap = [
0, 0, 1
0, 1, 0
1, 0, 0
1, 0, 1];

for k = 1:numel(fields)
    
    filename = fullfile(data_file_prefix, strcat(fields{k}, data_file_suffix));
    data = 100 * readmatrix(filename);
    
    aline(k) = stdshade(data, 0.2, cMap(k, :));
    aline(k).LineStyle = my_line_style{k};
    aline(k).Marker = my_marker{k};
    aline(k).LineWidth = 2;

    mean_data = mean(data);
    
    all_data = [all_data; mean_data];
    all_std = [all_std; std(data)];
    my_legend = [my_legend, al_strategies.(fields{k})];
     
end

hleg = legend(aline, my_legend, 'location', 'southeast', 'fontsize', 22);
set(hleg, 'box', 'off')

set(gca, 'fontsize', 32, 'ygrid', 'on', 'xgrid', 'on')
xlim([30, 90])
ylim([20, 100])
xlabel('Number of human-annotated samples')
ylabel('Accuracy (%)')
xticks(10:10:90)
yticks(20:10:100)
title('Plasma')

axes('Position', [0.6, 0.45, 0.25, 0.25])
box on, hold on

for i_row = 1:size(all_data, 1)
    data = all_data(i_row, 11:end);
    std_err = all_std(i_row, 11:end);
    indexOfInterest = (x >= 60) & (x <= 80);
    plot(x(indexOfInterest), data(indexOfInterest), line_spec{i_row}, 'linewidth', 2, 'color', cMap(i_row, :))
    % errorbar(x(indexOfInterest), 100 * data(indexOfInterest), 100 * std_err(indexOfInterest), line_spec{i_row}, 'linewidth', 2)
end
set(gca, 'fontsize', 18)
xticks([60:5:80])
ylim([84, 90])
yticks([84:2:90])
hold off

saveas(gcf, './Image/combine.png')
