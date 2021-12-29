% dataset analysis of EEM multi

clear, clc

eem_multi_dataset = '../data/2ET_40_trim20_L.xlsx';
data_eem_multi = readtable(eem_multi_dataset);


ex_data = data_eem_multi{:, 1};
em_data = data_eem_multi{:, 2};
group = [1*ones(40, 1); 2*ones(40, 1); 3*ones(40, 1); 4*ones(40, 1)];

data_matrix = data_eem_multi{:, 3:end};

X = unique(group);

count = zeros(numel(X), 1);

for i = 1:numel(X)
   count(i) =  sum(group == X(i)); 
end

disp('EEM category count')
X = X';
count = count';

title_arr = {'Phage infected E. coli', 'E. coli only', 'Phage only', 'Listeria'};
for i = 1:numel(X)
    i_sample = find(group == X(i));
    i_sample = i_sample(1);
    
    data_to_plot = data_matrix(:, i_sample);
    plot_matrix = zeros(max(ex_data), max(em_data));
    
    for i_value = 1:length(data_to_plot)
        plot_matrix(ex_data(i_value), em_data(i_value)) = data_to_plot(i_value);
    end
    
    figure(i), clf
    set(gcf, 'position', [500, 500, 800, 600])
    
    plot_matrix(~any(plot_matrix, 2), :) = [];
    plot_matrix(:, ~any(plot_matrix, 1)) = [];
    plot_matrix = log(plot_matrix);
    plot_matrix(isinf(plot_matrix)|isnan(plot_matrix)) = 0;
    plot_matrix = abs(plot_matrix);
    h = heatmap(plot_matrix);
    grid off
    xlabel('Excitation Wavelength (nm)')
    ylabel('Emission Wavelength (nm)')
    
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
    
    saveas(gcf, strcat('Image/eem_multi_sample_', num2str(i), '.png'))
end



