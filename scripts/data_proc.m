%% CLEAR WORK SPACE

clear; clc; close all;

%% LOAD DATA

coo_data_train = load('coo_data_train.txt');
coo_row_train = load('coo_row_train.txt');
coo_col_train = load('coo_col_train.txt');

coo_row_train = coo_row_train + 1;
coo_col_train = coo_col_train + 1;

m = max(unique(coo_row_train));
n = max(unique(coo_col_train));

train_data = sparse(coo_row_train, coo_col_train, coo_data_train, m, n);

[feature_train, eig_train] = my_sparse_pca(train_data, 50, 7000);

save ('train.mat', 'feature_train', 'eig_train');

clear;

coo_data_test = load('coo_data_test.txt');
coo_row_test = load('coo_row_test.txt');
coo_col_test = load('coo_col_test.txt');

coo_row_test = coo_row_test + 1;
coo_col_test = coo_col_test + 1;

m = max(unique(coo_row_test));
n = max(unique(coo_col_test));

test_data = sparse(coo_row_test, coo_col_test, coo_data_test, m, n);

[feature_test, eig_test] = my_sparse_pca(test_data, 50, 7000);

save ('test.mat', 'feature_test', 'eig_test');

clear; clc; close all;

%% STORE

load('test.mat')
load('train.mat')

csvwrite('feature_train.csv', feature_train);
csvwrite('feature_test.csv', feature_test);


%% POST PROCESS

dia_eigs_test = diag(eig_test);
dia_eigs_train = diag(eig_train);

sum_eig_test = sum(dia_eigs_test);
sum_eig_train = sum(dia_eigs_train);

for i = 1 : length(dia_eigs_test)

    curr_sum_test(i) = sum(dia_eigs_test(1:i)) / sum_eig_test;

end

for i = 1 : length(dia_eigs_train)

    curr_sum_train(i) = sum(dia_eigs_train(1:i)) / sum_eig_train;

end

figure;

subplot(1, 2, 1)

plot(1:length(dia_eigs_test), curr_sum_test, 'blue')
hold on
plot(1:length(dia_eigs_test), repmat(0.95, 1, length(dia_eigs_test)), 'red')
plot(1:length(dia_eigs_test), repmat(0.90, 1, length(dia_eigs_test)), 'green')
xlabel('# of eig values')
ylabel('energy percentage')
title('test data set')


subplot(1, 2, 2)

plot(1:length(dia_eigs_train), curr_sum_train, 'blue')
hold on
plot(1:length(dia_eigs_train), repmat(0.95, 1, length(dia_eigs_train)), 'red')
plot(1:length(dia_eigs_train), repmat(0.90, 1, length(dia_eigs_train)), 'green')

xlabel('# of eig values')
ylabel('energy percentage')
title('train data set')

