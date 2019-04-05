function [w_all, b_all] = task1_soft_margin_polynomial_kernel()
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
    C = [0.1, 0.6, 1.1, 2.1];
    lb = zeros(train_size, 1);
    x0 = [];
    options = optimset('LargeScale', 'off', 'MaxIter', 1000);
    p = [2 3 4 5];
    w_all = zeros(features, size(p, 2) * size(C, 2));
    b_all = zeros(1, size(p, 2) * size(C, 2));
    for C_idx = 1:size(C, 2)
        ub = ones(train_size, 1) .* C(C_idx);
        for p_idx = 1:size(p, 2)
            K = zeros(train_size, train_size);
            %H = zeros(train_size, train_size);
            for i = 1:train_size
                for j = 1:train_size
                    K(i, j) = (train_data(:, i)' * train_data(:, j) + 1) ^ p(p_idx);
                end
            end
            e = eig(K);
            if sum(e < 0) > 0
                disply(strcat('p = ', num2str(p), ' does not satisfy Mercer condition'));
                continue;
            end
            H = train_label .* train_label' .* K;
            alpha = quadprog(H, f, [], [], Aeq, beq, lb, ub, x0, options);
            sv_index = find(alpha <= C(C_idx));
            sv_size = size(sv_index, 1);
            b_sv = zeros(sv_size, 1);
            for i = 1:sv_size
                b_sv(i) = train_label(sv_index(i)) - sum(alpha .* train_label .* (train_data(sv_index(i))' * train_data + 1)' .^ p(p_idx));
            end
            b = sum(b_sv) / sv_size;
            
            save(strcat('task1_hard_margin_polynomial_kernel_', num2str(C(C_idx)) + '_' + num2str(p(p_idx)), '.mat'), 'w', 'b');
            w_all(:, (C_idx - 1) * size(p, 2) + p_idx) = w;
            b_all((C_idx - 1) * size(p, 2) + p_idx) = b;
        end
    end
end