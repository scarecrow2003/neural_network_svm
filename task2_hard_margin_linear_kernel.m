function [train_acc, test_acc] = task2_hard_margin_linear_kernel()
    train = load('train.mat');
    test = load('test.mat');
    train_size = size(train.train_data, 2);
    test_size = size(test.test_data, 2);
    data = [train.train_data test.test_data];
    predict = task1_hard_margin_linear_kernel(data);
    train_acc = sum(predict(1, 1:train_size) == train.train_label') / train_size;
    test_acc = sum(predict(1, train_size+1:train_size+test_size) == test.test_label') / test_size;
end