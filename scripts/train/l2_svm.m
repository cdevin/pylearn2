function theta = l2_svm(data_path, label_path, C, standard, train_num)

  load(data_path);
  load(label_path);

  trainXC = double(train_x);
  trainY = int32(train_y) + 1;
  clear train_x, train_y;
  testXC = double(test_x);
  testY = int32(test_y) + 1;
  clear test_x, test_y;
  C = str2num(C);
  train_num = double(train_num);


  %%% prepare data
  % train data
  % standardize data
  if strcmpi(standard,'True')
    trainXC_mean = mean(trainXC);
    trainXC_sd = sqrt(var(trainXC)+0.01);
    trainXC = bsxfun(@rdivide, bsxfun(@minus, trainXC, trainXC_mean), trainXC_sd);
  end
  trainXC = [trainXC, ones(size(trainXC,1),1)]; % intercept term

  % test data
  if strcmpi(standard, 'True')
    testXC = bsxfun(@rdivide, bsxfun(@minus, testXC, trainXC_mean), trainXC_sd);
  end
  testXC = [testXC, ones(size(testXC,1),1)];

  if size(C, 2) > 1
    train_x = trainXC(1:train_num,:);
    train_y = trainY(1:train_num);
    valid_x = trainXC(train_num:end,:);
    valid_y = trainY(train_num:end);
    best_valid_acc = -1;
    train_acc = -1;
    for i = 1:size(C, 2)
      fprintf('C value:%f\n', C(i))
      % train classifier using SVM
      theta = train_svm(train_x, train_y, C(i));
      [val,labels] = max(valid_x*theta, [], 2);
      acc = 100 * (1 - sum(labels ~= valid_y) / length(valid_y));
      if acc > best_valid_acc
         best_C = C(i);
         best_valid_acc = acc;
         [val,labels] = max(train_x*theta, [], 2);
         train_acc = 100 * (1 - sum(labels ~= train_y) / length(train_y));
      end
    end
  else
    best_C = C(1);
    theta = train_svm(trainXC, trainY, best_C);
    [val,labels] = max(trainXC*theta, [], 2);
    train_acc = 100 * (1 - sum(labels ~= trainY) / length(trainY));
  end


  % train classifier using SVM
  theta = train_svm(trainXC, trainY, best_C);

  fprintf('Best_C was: %f\n', best_C);
  fprintf('Train_accuracy %f%%\n', train_acc);
  fprintf('Valid_accuracy %f%%\n', best_valid_acc);

  %%%%% TESTING %%%%%

    % test and print result
  [val,labels] = max(testXC*theta, [], 2);
  fprintf('Test_accuracy %f%%\n', 100 * (1 - sum(labels ~= testY) / length(testY)));


