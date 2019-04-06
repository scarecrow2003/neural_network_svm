function C_value = task3()
    p = 1;
    C = 1:0.01:5;
    train = load('train.mat');
    test = load('test.mat');
    train_size = size(train.train_data, 2);
    test_size = size(test.test_data, 2);
    data = [train.train_data test.test_data];
    % standardize the training data
    mean_value = mean(train.train_data, 2);
    sd = std(train.train_data, 0, 2);
    train_data = (train.train_data - mean_value) ./ sd;
    
    train_label = train.train_label;
    train_size = size(train_data, 2);
    % calculate K
    K = zeros(train_size, train_size);
    for i = 1:train_size
        for j = 1:train_size
            K(i, j) = (train_data(:, i)' * train_data(:, j) + 1) ^ p;
        end
    end
    e = eig(K);
    if sum(e < -1e-4) > 0
        display(strcat('p = ', num2str(p), ' does not satisfy Mercer condition'));
    end
    % calculate H from K
    options = optimset('LargeScale', 'off', 'MaxIter', 1000);
    max_test_acc = 0;
    C_index = 0;
    result_alpha = zeros(train_size, 1);
    result_p = p;
    result_b = 0;
    for k = 1:size(C, 2)
        display(strcat('C = ', num2str(C(k))));
        H = train_label .* train_label' .* K;
        f = -ones(train_size, 1);
        Aeq = train_label';
        beq = 0;
        lb = zeros(train_size, 1);
        x0 = [];
        ub = ones(train_size, 1) .* C(k);
        % calculate alpha
        alpha = quadprog(H, f, [], [], Aeq, beq, lb, ub, x0, options);
        % get support vector index
        sv_index = find(alpha <= C(k) & alpha > 1e-4);
        sv_size = size(sv_index, 1);
        b_sv = zeros(sv_size, 1);
        for i = 1:sv_size
            b_sv(i) = train_label(sv_index(i)) - sum(alpha .* train_label .* (train_data(:, sv_index(i))' * train_data + 1)' .^ p);
        end
        % use support vectors to calculate b
        b = sum(b_sv) / sv_size;
    
        % standardize the data
        standardize_data = (data - mean_value) ./ sd;
        % calculate g(x)
        gx = sum(alpha .* train_label .* (train_data' * standardize_data + 1) .^ p, 1) + b;
        predict = zeros(1, size(gx, 2));
        predict(gx > 0) = 1;
        predict(gx <= 0) = -1;
        train_acc = sum(predict(1, 1:train_size) == train.train_label') / train_size;
        test_acc = sum(predict(1, train_size+1:train_size+test_size) == test.test_label') / test_size;
        display(strcat('train_acc:', num2str(train_acc)));
        display(strcat('test_acc:', num2str(test_acc)));
        if test_acc > max_test_acc
            max_test_acc = test_acc;
            C_index = k;
            result_alpha = alpha;
            result_b = b;
        end
    end
    C_value = C(C_index);
    save('task3_svm.mat', 'result_alpha', 'result_p', 'result_b', 'train_data', 'train_label', 'mean_value', 'sd');
end