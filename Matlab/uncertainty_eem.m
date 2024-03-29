% compare

clear, clc

my_line_style = {':', '-', '-', '-.', '-'};
my_marker = {'none', 'o', '+', 'none', '^'};
line_spec = {':', 'o-', '+-', '-.', '^-'};

% al = Active Learning

al_strategies = struct( ...
    'random_eem', 'Baseline', ...
     'uncertainty_leastConfident_eem', 'Least Confident', ...
     'uncertainty_entropy_eem', 'Entropy');

data_file_prefix = '../Python/Result/';
data_file_suffix = '.csv';


fields = fieldnames(al_strategies);

my_legend = {};
all_data = [];
all_std = [];


figure(1), clf, hold on
set(gcf, 'position', [500, 500, 1000, 650])

x = 11:56;

cMap = lines(length(fields));


for k = 1:numel(fields)
    
    filename = fullfile(data_file_prefix, strcat(fields{k}, data_file_suffix));
    data = 100 * readmatrix(filename);
    
    aline(k) = stdshade(data, 0.2, cMap(k, :));
    aline(k).LineStyle = my_line_style{k};
    aline(k).Marker = my_marker{k};
    aline(k).LineWidth = 5;

    mean_data = mean(data);
    
    all_data = [all_data; mean_data];   
    all_std = [all_std; std(data)];
    my_legend = [my_legend, al_strategies.(fields{k})];
     
end

hleg = legend(aline, my_legend, 'location', 'southeast');
%set(hleg, 'box', 'off')

set(gca, 'fontsize', 32, 'ygrid', 'on', 'xgrid', 'on')
xlim([25, 56])
ylim([50, 90])
xlabel('Number of labeled samples')
ylabel('Accuracy (%)')
xticks(10:10:90)
yticks(20:10:100)
title('Uncertainty Sampling')
hold off

saveas(gcf, './Image/uncertainty_eem.png')
