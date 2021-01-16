%% Generate Data

clear all, close all, clc

plotData = 1;
n = 2; Ntrain = 1000; Ntest = 10000; 
alpha = [0.33,0.34,0.33]; % must add to 1.0
meanVectors = [-18 0 18;-8 0 8];
covEvalues = [3.2^2 0;0 0.6^2];
covEvectors(:,:,1) = [1 -1;1 1]/sqrt(2);
covEvectors(:,:,2) = [1 0;0 1];
covEvectors(:,:,3) = [1 -1;1 1]/sqrt(2);

t = rand(1,Ntrain);
ind1 = find(0 <= t & t <= alpha(1));
ind2 = find(alpha(1) < t & t <= alpha(1)+alpha(2));
ind3 = find(alpha(1)+alpha(2) <= t & t <= 1);
Xtrain = zeros(n,Ntrain);
Xtrain(:,ind1) = covEvectors(:,:,1)*covEvalues^(1/2)*randn(n,length(ind1))+meanVectors(:,1);
Xtrain(:,ind2) = covEvectors(:,:,2)*covEvalues^(1/2)*randn(n,length(ind2))+meanVectors(:,2);
Xtrain(:,ind3) = covEvectors(:,:,3)*covEvalues^(1/2)*randn(n,length(ind3))+meanVectors(:,3);

t = rand(1,Ntest);
ind1 = find(0 <= t & t <= alpha(1));
ind2 = find(alpha(1) < t & t <= alpha(1)+alpha(2));
ind3 = find(alpha(1)+alpha(2) <= t & t <= 1);
Xtest = zeros(n,Ntrain);
Xtest(:,ind1) = covEvectors(:,:,1)*covEvalues^(1/2)*randn(n,length(ind1))+meanVectors(:,1);
Xtest(:,ind2) = covEvectors(:,:,2)*covEvalues^(1/2)*randn(n,length(ind2))+meanVectors(:,2);
Xtest(:,ind3) = covEvectors(:,:,3)*covEvalues^(1/2)*randn(n,length(ind3))+meanVectors(:,3);

if plotData == 1
    figure(1), subplot(1,2,1),
    plot(Xtrain(1,:),Xtrain(2,:),'.')
    title('Training Data'), axis equal,
    subplot(1,2,2),
    plot(Xtest(1,:),Xtest(2,:),'.')
    title('Testing Data'), axis equal,
end


%%
data = Xtrain(1,:);
label = Xtrain(2,:);
indices = crossvalind('Kfold',Xtrain(2,:),10);
%%
tsfn_fun = ["logsig","poslin"];
%hidden_units = [2 5 7 10 25 50 100];
hidden_units = [2 5 10 15 25 50];
for i = 1:length(tsfn_fun)
    tsfn_ = tsfn_fun(i);
    for j = 1:length(hidden_units)
        hidden = hidden_units(j);
        net = define_NN(tsfn_,hidden);
        mean_cv = 0;
        for k = 1:10
            test_data = (indices == k); 
            train_data = ~test_data;
            trained_net = train(net,data(train_data),label(train_data));
            y_pred = trained_net(data(test_data));
            MSE = perform(trained_net,y_pred,label(test_data));
            cv(k) = MSE;
            if k == 10
                mean_cv = mean(cv);
            end
        end
        metrics(i,j) = mean_cv;
        fprintf("Transfer Fun %s - # Hidden Units %d - MSE %f\n",tsfn_,hidden,mean_cv)
        pause(0.00001)
    end
end
%%
[minimum,idx] = min(metrics,[],'all','linear');
[act_idx,hidden_idx] = ind2sub(size(metrics),idx);
transfer_fun = tsfn_fun(act_idx);
hidden_unit = hidden_units(hidden_idx);
fprintf("Best TSFN Function %s - Best # Hidden Units %d\n",transfer_fun,hidden_unit)
%%
net = define_NN(transfer_fun,hidden_unit);
trained_net = train(net,data,label);

y_pred = trained_net(Xtest(1,:));
figure(1)
subplot(1,2,2)
hold on
plot(Xtest(1,:),y_pred,'r*')
MSE_FINAL = perform(net,y_pred,Xtest(2,:))


%% Show validation results
figure
i = 1:length(hidden_units);
bar(i,metrics','grouped');
g = gca;
xticklabels = {'2','5','10','15','25','50'}';
g.XTickLabel = xticklabels;
legend(["Feature 1","Feature 2",]);
ylabel("Mean Square Error")
title("MSE vs # of Hidden Nuerons")






