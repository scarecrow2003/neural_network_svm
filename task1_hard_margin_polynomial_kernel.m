function [w_all, b_all] = task1_hard_margin_polynomial_kernel()
    train = load('train.mat');
    mean_value = mean(train.train_data, 2);
    sd = std(train.train_data, 0, 2);
    train_data = (train.train_data - mean_value) ./ sd;
    train_label = train.train_label;
    train_size = size(train_data, 2);
    features = size(train_data, 1);
    f = -ones(train_size, 1);
    Aeq = train_label';
    beq = 0;
    C = 1.0e6;
    lb = zeros(train_size, 1);
    ub = ones(train_size, 1) .* C;
    x0 = [];
    options = optimset('LargeScale', 'off', 'MaxIter', 1000);
    p = [2 3 4 5];
    w_all = zeros(features, size(p, 2));
    b_all = zeros(1, size(p, 2));
    for p_idx = 1:size(p, 2)
        H = zeros(train_size, train_size);
        for i = 1:train_size
            for j = 1:train_size
                H(i, j) = train_label(i) * train_label(j) * (train_data(:, i)' * train_data(:, j) + 1) ^ p(p_idx);
            end
        end
        alpha = quadprog(H, f, [], [], Aeq, beq, lb, ub, x0, options);
        w = (sum(alpha .* train_label .* train_data', 1))';
        sv_index = 0;
        for i = 1:train_size
            if alpha(i) > 1e-4
                sv_index = i;
                break;
            end
        end
        b = 1 / train_label(sv_index) - w' * train_data(:, sv_index);
        save(strcat('task1_hard_margin_polynomial_kernel_', num2str(p(p_idx)), '.mat'), 'w', 'b');
        w_all(:, p_idx) = w;
        b_all(p_idx) = b;
    end
end