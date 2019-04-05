function [train_acc, test_acc] = task2_hard_margin_linear_kernel()
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
    svm = load('task1_hard_margin_linear_kernel.mat');
    train_predict = svm.w' * train_data + svm.b;
    train_predict(train_predict >= 0) = 1;
    train_predict(train_predict < 0) = -1;
    train_acc = sum(train_predict == train_label) / train_size;
    test_predict = svm.w' * test_data + svm.b;
    test_predict(test_predict >= 0) = 1;
    test_predict(test_predict < 0) = -1;
    test_acc = sum(test_predict == test_label) / test_size;
end