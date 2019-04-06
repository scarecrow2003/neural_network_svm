function [train_acc, test_acc] = task2_hard_margin_polynomial_kernel()
    train = load('train.mat');
    test = load('test.mat');
    train_size = size(train.train_data, 2);
    test_size = size(test.test_data, 2);
    data = [train.train_data test.test_data];
    p = [2 3 4 5];
    p_size = size(p, 2);
    train_acc = zeros(1, p_size);
    test_acc = zeros(1, p_size);
    for i = 1:p_size
        display('p = ' + p(i));
        predict = task1_hard_margin_polynomial_kernel(data, p(i));
        train_acc(i) = sum(predict(1, 1:train_size) == train.train_label') / train_size;
        test_acc(i) = sum(predict(1, train_size+1:train_size+test_size) == test.test_label') / test_size;
    end
end