%% Generat Data Question 2
clear all; close all; clc

% Generate samples from a 4-component GMM
alpha_true = [0.1,0.2,0.3,.4]; %priors
mu_true = [-8 0 8 0;0 0 0 9; 2 3 2 3]';
Sigma_true(:,:,1) = [7 1 1;1 8 1; 1 1 4];
Sigma_true(:,:,2) = [7 1 1;1 2 1; 1 1 3];
Sigma_true(:,:,3) = [4 1 1;1 8 1; 1 1 2];
Sigma_true(:,:,4) = [4 1 1;1 8 1; 1 1 5];


%Creat GMM
gm = gmdistribution(mu_true, Sigma_true, alpha_true);
%rng('default');
N = 1000;
[Y, compIdx] = random(gm,N);

% True GMM visualization
figure(1)
scatter3(Y(:,1),Y(:,2),Y(:,3),3,compIdx);
title('GMM');
xlabel('Feature 1')
ylabel('Feature 2')
zlabel('Feature 3')


% decision rule is decide class j if p(c_j | x) > p(c_k | x) and  p(c_l | x) for all j
pc1_x = mvnpdf(Y, mu_true(1,:), Sigma_true(:,:,1)) * alpha_true(1);
pc2_x = mvnpdf(Y, mu_true(2,:), Sigma_true(:,:,2)) * alpha_true(2);
pc3_x = mvnpdf(Y, mu_true(3,:), Sigma_true(:,:,3)) * alpha_true(3);
pc4_x = mvnpdf(Y, mu_true(4,:), Sigma_true(:,:,3)) * alpha_true(4);
[~,predictions] = max([pc1_x,pc2_x,pc3_x,pc4_x],[],2);


%% a
tabulate(compIdx);
C = confusionmat(compIdx,predictions);
figure(2)
confusionchart(C)
sum(compIdx~=predictions);
sum(compIdx~=predictions)/length(Y);
%% Function to plot MAP estimate errors
figure(3)

data = Y;
classIndex =  compIdx;
nClass = 4;
my_inference = predictions;

markerStrings = 'xo.+';
errors = my_inference~=classIndex;
legendStrings = [];
for idxClass = 1:nClass
    idx_correct = (~errors) & (classIndex==idxClass);
    dataClass = data(idx_correct,:);
    scatter3(dataClass(:,1), dataClass(:,2) ,dataClass(:,3));
    hold on
    if ~isempty(dataClass)
        legendStrings = [legendStrings "Class "+num2str(idxClass)+" Correct"];
    end
    
    idx_incorrect = (errors) & (classIndex==idxClass);
    dataClass = data(idx_incorrect,:);
    scatter3(dataClass(:,1), dataClass(:,2) , dataClass(:,3), 'MarkerEdgeColor','k','MarkerFaceColor',[1 1 0] );
    hold on
    if ~isempty(dataClass)
        legendStrings = [legendStrings "Class "+num2str(idxClass)+" Wrong"];
    end
    
    %
end
xlabel('Feature 1');
ylabel('Feature 2');
zlabel('Feature 3');

legend(legendStrings);
title("MAP CLASSIFIER | Pe = "+num2str(1-(1-sum(errors)/length(data)))) 

%% Part 3

%Creat GMM 100, 1000, 10000 Samples
gm_100 = gmdistribution(mu_true, Sigma_true, alpha_true);
gm_1000 = gmdistribution(mu_true, Sigma_true, alpha_true);
gm_10000 = gmdistribution(mu_true, Sigma_true, alpha_true);

[Y_100, compIdx_100] = random(gm,100);
[Y_1000, compIdx_1000] = random(gm,1000);
[Y_10000, compIdx_10000] = random(gm,10000);

% True GMM visualization
figure(4)
scatter3(Y_100(:,1),Y_100(:,2),Y_100(:,3),3,compIdx_100);
title('GMM 100'); xlabel('Feature 1'); ylabel('Feature 2'); zlabel('Feature 3')

figure(5)
scatter3(Y_1000(:,1),Y_1000(:,2),Y_1000(:,3),3,compIdx_1000);
title('GMM 1000'); xlabel('Feature 1'); ylabel('Feature 2'); zlabel('Feature 3')

figure(6)
scatter3(Y_10000(:,1),Y_10000(:,2),Y_10000(:,3),3,compIdx_10000);
title('GMM 10000'); xlabel('Feature 1'); ylabel('Feature 2'); zlabel('Feature 3')


%% Organize data
data = cell(1,3);
data{1} = Y_100; 
data{2} = Y_1000; 
data{3} = Y_10000;

label = cell(1,3)
label{1} = compIdx_100;
label{2} = compIdx_1000;
label{3} = compIdx_10000;

%% K fold cross validation

for sets = 1:3
    indices = crossvalind('Kfold',label{sets},10);
    tsfn_fun = "logsig";
    hidden_units = [2 5 10 25 50 100];
    dataset = data{sets};
    datalabels = label{sets};
    for j = 1:length(hidden_units)
        hidden = hidden_units(j);
        net = setupNN(tsfn_fun,hidden);
        mean_cv = 0;
        for k = 1:10
            test_data = (indices == k); 
            train_data = ~test_data;
            trained_net = train(net,dataset(train_data,:)',ind2vec(datalabels(train_data)',4));
            y_pred = trained_net(dataset(test_data,:)');
            CE = perform(trained_net,ind2vec(datalabels(test_data)',4),y_pred);
            cv(k) = CE;
            if k == 10
                mean_cv = mean(cv);
            end
        end
        metrics(sets,j) = mean_cv;
        fprintf("Set %d - # Hidden Units %d - CE %f\n",sets,hidden,mean_cv)
        pause(0.00001)
    end
end 


%% Find best Dataset and # nuerons
[minimum,idx] = min(metrics,[],'all','linear');
[set,hidden_idx] = ind2sub(size(metrics),idx);
hidden_unit = hidden_units(hidden_idx);
fprintf("Best Dataset %d - Best # Hidden Units %d\n",set,hidden_unit)

%% Apply trained neural network to test dataset 
net = setupNN(tsfn_fun,hidden_unit);
dataset = data{set}';
datasetlabel = label{set}';
trained_net = train(net,dataset,ind2vec(datasetlabel,4));

y_pred = trained_net(Y');
perf = perform(trained_net,ind2vec(compIdx',4),y_pred);
classes = vec2ind(y_pred);

%% P(error)
p_error = sum(classes'~=compIdx)/N
%% Show validation results
figure
i = 1:length(hidden_units);
bar(i,metrics','grouped');
g = gca;
xticklabels = {'2','5','10','25','50','100'}';
g.XTickLabel = xticklabels;
legend(["100","1000","10000"]);
ylabel("Cross Entropy")
title("Cross Entropy vs # of Hidden Nuerons")
