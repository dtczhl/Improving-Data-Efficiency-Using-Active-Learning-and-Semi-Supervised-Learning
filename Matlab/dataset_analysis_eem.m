% dataset analysis of EEM

clear, clc

eem_dataset = '../data/Processed/eem_dataset.csv';

ex_em = '../data/Processed/eem_ex_em.csv';
ex_em_data = dlmread(ex_em);

em_data = mod(ex_em_data, 1000);
ex_data = floor(ex_em_data/1000);

line_style = {'-', '-', '-', '-'};

data_eem = readtable(eem_dataset);

group = data_eem(2:end, end);
group = group{:, :};

data_matrix = data_eem(2:end, 1:end-1);
data_matrix = data_matrix{:, :};
data_matrix = fliplr(data_matrix);

X = unique(group);

count = zeros(numel(X), 1);

for i = 1:numel(X)
   count(i) =  sum(group == X(i)); 
end

disp('EEM category count')
X = X';
count = count';


figure(1), clf, hold on
set(gcf, 'position', [500, 500, 800, 500])
barh([1], [count(3)], 0.5)
barh([0], [count(1);count(2)], 0.5, 'stacked')
set(gca, 'xgrid', 'on', 'fontsize', 30)
xlabel('Number of samples')
ylabel('EEM')
legend({'T7 infected E. coli', 'T7 only', 'E. coli only'})
xlim([0, 60])
ylim([-0.5, 3])
yticks([0, 1])
yticklabels({'Control', 'Target'})
saveas(gcf, 'Image/eem_count.png')

title_arr = {'T7 infected E. coli', 'T7 only', 'E. coli only'};
for i = 1:numel(X)
    i_sample = find(group == X(i));
    i_sample = i_sample(1);
    
    data_to_plot = data_matrix(i_sample, :);
    plot_matrix = zeros(max(ex_data), max(em_data));
    
    for i_value = 1:length(data_to_plot)
        plot_matrix(ex_data(i_value), em_data(i_value)) = data_to_plot(i_value);
    end
    
    figure(1+i), clf
    set(gcf, 'position', [500, 500, 800, 600])
    
    plot_matrix(~any(plot_matrix, 2), :) = [];
    plot_matrix(:, ~any(plot_matrix, 1)) = [];
    plot_matrix = log(plot_matrix);
    plot_matrix(isinf(plot_matrix)|isnan(plot_matrix)) = 0;
    plot_matrix = abs(plot_matrix);
    h = heatmap(plot_matrix);
    grid off
    xlabel('EX (Hz)')
    ylabel('EM (Hz)')
    
    xticks = unique(em_data);
    customxlabels = string(xticks);
    for i_xtick = 1:length(xticks)
        if mod(i_xtick, 5) ~= 0
            customxlabels(i_xtick) = '';
        end
    end
    h.XDisplayLabels = customxlabels;
    
    yticks = unique(ex_data);
    customylabels = string(yticks);
    for i_ytick = 1:length(yticks)
        if mod(i_ytick, 5) ~= 0
            customylabels(i_ytick) = '';
        end
    end
    h.YDisplayLabels = customylabels;
    
    set(gca, 'fontsize', 24)
    title(title_arr{i})
    
    saveas(gcf, strcat('Image/eem_sample_', num2str(i), '.png'))
end








