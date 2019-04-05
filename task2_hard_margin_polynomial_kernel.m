function [train_acc, test_acc] = task2_hard_margin_polynomial_kernel()
    train = load('train.mat');
    mean_value = mean(train.train_data, 2);
    sd = std(train.train_data, 0, 2);
    train_data = (train.train_data - mean_value) ./ sd;
    train_label = train.train_label';
    train_size = size(train_data, 2);
    test = load('test.mat');
    test_data = (test.test_data - mean_value) ./ sd;
    test_label = test.test_label';
    test_size = size(test_data, 2);
    p = [2 3 4 5];
    p_size = size(p, 2);
    train_acc = zeros(1, p_size);
    test_acc = zeros(1, p_size);
    for i = 1:p_size
        svm = load(strcat('task1_hard_margin_polynomial_kernel_', num2str(p(i)), '.mat'));
        train_predict = svm.w' * train_data + svm.b;
        train_predict(train_predict >= 0) = 1;
        train_predict(train_predict < 0) = -1;
        train_acc(i) = sum(train_predict == train_label) / train_size;
        test_predict = svm.w' * test_data + svm.b;
        test_predict(test_predict >= 0) = 1;
        test_predict(test_predict < 0) = -1;
        test_acc(i) = sum(test_predict == test_label) / test_size;
    end
end