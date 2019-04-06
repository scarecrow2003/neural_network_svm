function label = task1_hard_margin_linear_kernel(data)
    train = load('train.mat');
    % standardize the training data
    mean_value = mean(train.train_data, 2);
    sd = std(train.train_data, 0, 2);
    train_data = (train.train_data - mean_value) ./ sd;

    train_label = train.train_label;
    train_size = size(train_data, 2);
    % calculate H
    H = zeros(train_size, train_size);
    for i = 1:train_size
        for j = 1:train_size
            H(i, j) = train_label(i) * train_label(j) * train_data(:, i)' * train_data(:, j);
        end
    end
    f = -ones(train_size, 1);
    Aeq = train_label';
    beq = 0;
    C = 1.0e6;
    lb = zeros(train_size, 1);
    ub = ones(train_size, 1) .* C;
    x0 = [];
    options = optimset('LargeScale', 'off', 'MaxIter', 1000);
    % calculate alpha
    alpha = quadprog(H, f, [], [], Aeq, beq, lb, ub, x0, options);
    % calcuate w
    w = (sum(alpha .* train_label .* train_data', 1))';
    % get a support vector index
    [~, sv_index] = max(alpha);
    % use support vector to calculate b
    b = 1 / train_label(sv_index) - w' * train_data(:, sv_index);
    
    % standardize the data
    standardize_data = (data - mean_value) ./ sd;
    % calculate g(x)
    gx = w' * standardize_data + b;
    label = zeros(1, size(gx, 2));
    label(gx > 0) = 1;
    label(gx <= 0) = -1;
end