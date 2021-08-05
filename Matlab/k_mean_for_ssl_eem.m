% EEM k mean for semi-supervised learning

clear, clc


% [1, 0, 2, 0]
k_mean_label = ...
    [0 1 1 1 0 0 1 1 1 0 1 1 1 0 0 1 0 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 ...
    1 0 1 1 1 1 1 0 1 1 1 0 0 0 0 1 1 1 0 0 1 0 1 1 0 0 1 1 0 0 0 0 1 0 1];

counts = [48 24];
label = [1, 0];
label_vector = [label(1) * ones(1, counts(1)), label(2) * ones(1, counts(2))];


a1 = sum(k_mean_label(1:48) == label(1));
a2 = sum(k_mean_label(48+1:48+24) == label(2));

[a1, a2]

(1 - (a1+a2) / sum(counts)) * 100

confuse_matrix = confusionmat(label_vector, k_mean_label);
confuse_matrix(1, :) = confuse_matrix(1, :) / sum(confuse_matrix(1, :));
confuse_matrix(2, :) = confuse_matrix(2, :) / sum(confuse_matrix(2, :));
confuse_matrix = confuse_matrix * 100;

figure(2), clf
set(gcf, 'position', [500, 500, 800, 600])
h = heatmap(confuse_matrix);
xlabel('Predicted Class')
ylabel('Actual Class')
set(gca, 'fontsize', 24)
caxis([0, 100]);
title('EEM')
saveas(gcf, 'Image/k_mean_eem.png')


