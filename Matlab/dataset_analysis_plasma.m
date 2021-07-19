% dataset analysis

clear, clc

plasma_dataset = '../data/Processed/plasma_dataset.csv';

line_style = {'-', '-', '-', '-'};

data_plasma = readtable(plasma_dataset);

fid = fopen(plasma_dataset, 'r');
header = textscan(fid, '%s', 1);
header = string(header);
header = strsplit(header, ',');

header_freq = zeros(1, numel(header)-1);
for i =1:length(header_freq)
   header_freq(i) = round(str2num(header(i)));
end

group = data_plasma(2:end, end);
group = group{:, :};

data_matrix = data_plasma(2:end, 1:end-1);
data_matrix = data_matrix{:, :};

X = unique(group);

count = zeros(numel(X), 1);

for i = 1:numel(X)
   count(i) =  sum(group == X(i)); 
end

disp('Plasma category count')
X = X';
count = count';

figure(1)
set(gcf, 'position', [500, 500, 800, 500])
barh(count, 0.5)
set(gca, 'xgrid', 'on', 'fontsize', 30)
xlabel('Number of samples')
ylabel('Plasma Dosage')
saveas(gcf, 'Image/plasma_count.png')

figure(2), clf, hold on
set(gcf, 'position', [500, 500, 1000, 600])
for i = 1:numel(X)
    i_sample = find(group == X(i));
    i_sample = i_sample(3);
    plot(header_freq, data_matrix(i_sample, :), line_style{i}, 'linewidth', 3)
end
legend({'A sample of dosage 1', 'A sample of dosage 2', 'A sample of dosage 3', 'A sample of dosage 4'}, 'location', 'northeast')
set(gca, 'fontsize', 30, 'yscale', 'linear')
xlabel('Frequency (Hz)')
ylabel('Intensity')
grid on
hold off
saveas(gcf, 'Image/plasma_sample.png')





