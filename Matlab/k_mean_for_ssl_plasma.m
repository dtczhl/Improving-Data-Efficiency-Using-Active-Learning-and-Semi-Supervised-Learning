% plasma k mean for semi-supervised learning

clear, clc


% [1, 0, 2, 0]
k_mean_label = ...
    [3 2 0 0 1 0 0 2 0 0 0 3 0 0 0 0 0 1 0 1 2 0 0 0 2 1 0 2 2 2 2 2 2 2 2 2 2 ...
    2 2 2 2 2 2 2 2 2 3 2 3 2 3 2 3 2 1 1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 ...
    1 1 1 1 1 1 1 1 1 1 1 2 3 1 2 2 1 2 2 2 1 1 2 2 2 1 2 3 1 2 2 1 2 2 2 3 2 ...
    2 1 1];

counts = [27, 27, 30, 30];
label = [0, 2, 1, 3];
label_vector = [label(1) * ones(1, counts(1)), label(2) * ones(1, counts(2)), label(3) * ones(1, counts(3)), label(4) * ones(1, counts(4))];


a1 = sum(k_mean_label(1:27) == label(1));
a2 = sum(k_mean_label(27+1:27+27) == label(2));
a3 = sum(k_mean_label(27+27+1:27+27+30) == label(3));
a4 = sum(k_mean_label(27+27+30+1:27+27+30+30) == label(4));

[a1, a2, a3, a4]

(1 - (a1+a2+a3+a4) / sum(counts)) * 100


confuse_matrix = confusionmat(label_vector, k_mean_label);
confuse_matrix(1, :) = confuse_matrix(1, :) / sum(confuse_matrix(1, :));
confuse_matrix(2, :) = confuse_matrix(2, :) / sum(confuse_matrix(2, :));
confuse_matrix(3, :) = confuse_matrix(3, :) / sum(confuse_matrix(3, :));
confuse_matrix(4, :) = confuse_matrix(4, :) / sum(confuse_matrix(4, :));
confuse_matrix = confuse_matrix * 100;

figure(1), clf
set(gcf, 'position', [500, 500, 800, 600])
h = heatmap(confuse_matrix);
xlabel('Predicted Class')
ylabel('Actual Class')
set(gca, 'fontsize', 24)
caxis([0, 100]);
title('Plasma')
saveas(gcf, 'Image/k_mean_plasma.png')

