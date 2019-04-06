function [train_acc, test_acc] = task2_soft_margin_polynomial_kernel()
    train = load('train.mat');
    test = load('test.mat');
    train_size = size(train.train_data, 2);
    test_size = size(test.test_data, 2);
    data = [train.train_data test.test_data];
    p = [1 2 3 4 5];
    C = [0.1 0.6 1.1 2.1];
    p_size = size(p, 2);
    C_size = size(C, 2);
    train_acc = zeros(p_size, C_size);
    test_acc = zeros(p_size, C_size);
    for i = 1:p_size
        for j = 1:C_size
            display(strcat('p = ', num2str(p(i)), ' C = ', num2str(C(j))));
            predict = task1_soft_margin_polynomial_kernel(data, p(i), C(j));
            train_acc(i, j) = sum(predict(1, 1:train_size) == train.train_label') / train_size;
            test_acc(i, j) = sum(predict(1, train_size+1:train_size+test_size) == test.test_label') / test_size;
        end
    end
end