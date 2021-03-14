% query by disagreement


clear, clc

data_folder = '../Save/1/';

line_spec = {'.-', 'o-', ':', '-.'};

threshold_arr = 0:4;

overall_accuracy_result = [];
overall_keep_accuracy = [];
overall_keep_size = [];


for thres = threshold_arr
    accuracy_file = fullfile(data_folder, strcat('query_by_disagreement_result_', num2str(thres), '.csv'));
    accuracy_data = readmatrix(accuracy_file);
    accuracy_data = mean(accuracy_data);
    overall_accuracy_result = [overall_accuracy_result; accuracy_data];
    
    keep_accuracy_file = fullfile(data_folder, strcat('query_by_disagreement_keep_accuracy_', num2str(thres), '.csv'));
    keep_accuracy_data = readmatrix(keep_accuracy_file)';
    keep_accuracy_data = reshape(keep_accuracy_data, 90, numel(keep_accuracy_data)/90);
    keep_accuracy_data = mean(keep_accuracy_data');
    overall_keep_accuracy = [overall_keep_accuracy; keep_accuracy_data];
    
    keep_size_file = fullfile(data_folder, strcat('query_by_disagreement_keep_size_', num2str(thres), '.csv'));
    keep_size_data = readmatrix(keep_size_file)';
    keep_size_data = reshape(keep_size_data, 90, numel(keep_size_data)/90);
    keep_size_data = mean(keep_size_data');
    overall_keep_size = [overall_keep_size; keep_size_data];

end


figure(1), clf, hold on
set(gcf, 'position', [500, 500, 1000, 600])
plot(overall_keep_size(1, 11:end), 100*overall_accuracy_result(1, 11:end), line_spec{1}, 'linewidth', 3)
plot(overall_keep_size(2, 11:end), 100*overall_accuracy_result(2, 11:end), line_spec{2}, 'linewidth', 3)
% plot(overall_keep_size(3, 11:end), 100*overall_accuracy_result(3, 11:end), line_spec{3}, 'linewidth', 2)
% plot(overall_keep_size(4, 11:end), 100*overall_accuracy_result(4, 11:end), line_spec{4}, 'linewidth', 2)
xlim([0, 90])
xticks([0:10:90])
xlabel('Number of labeled samples')
ylabel('Accuracy (%)')
% yticks([20:10:100])
legend('Passive', 'Query by disagreement', 'location', 'southeast')
% legend('No discarding', 'Discard samples of more than 1 label', 'Discard samples of mroe than 2 labels', 'Discard sampels of more than 3 labels', 'location', 'southeast')
set(gca, 'fontsize', 24, 'ygrid', 'on')
hold off
saveas(gcf, './Image/query_by_disagreement_accuracy.png')

figure(2), clf, hold on
set(gcf, 'position', [500, 500, 1000, 600])
plot([11:90], overall_keep_size(1, 11:end), line_spec{1}, 'linewidth', 3)
plot([11:90], overall_keep_size(2, 11:end), line_spec{2}, 'linewidth', 3)
%plot([11:90], overall_keep_size(3, 11:end), line_spec{3}, 'linewidth', 2)
%plot([11:90], overall_keep_size(4, 11:end), line_spec{4}, 'linewidth', 2)
xlim([0, 90])
xticks([0:10:90])
% yticks([0:10:90])
% ylim([10, 90])
xlabel('Number of streaming samples')
ylabel('Number of kept samples')
legend('Passive', 'Query by disagreement', 'location', 'southeast')
set(gca, 'fontsize', 24, 'ygrid', 'on')
hold off
saveas(gcf, './Image/query_by_disagreement_size.png')



% figure(1)
% plot([11:90], overall_accuracy_result(:, 11:end))
% xlim([0, 90])
% 
% figure(2)
% plot([11:90], overall_keep_size(:, 11:end))
% xlim([0, 90])
