s = load('task3_svm.mat');
% standardize the data
standardize_data = (eval_data - s.mean_value) ./ s.sd;
% calculate g(x)
gx = sum(s.result_alpha .* s.train_label .* (s.train_data' * standardize_data + 1) .^ s.result_p, 1) + s.result_b;
eval_predicted = zeros(1, size(gx, 2));
eval_predicted(gx > 0) = 1;
eval_predicted(gx <= 0) = -1;
acc = sum(eval_predicted == eval_label) / size(eval_label, 2);
display(strcat('acc: ', num2str(acc)));