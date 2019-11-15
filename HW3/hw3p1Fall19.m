%% Homeowork 3 Part 1

close all, clear all; clc;
delta = 1e-5; % tolerance for EM stopping criterion
regWeight = 1e-10; % regularization parameter for covariance estimates

% Generate samples from a 4-component GMM
alpha_true = [0.2,0.3,0.4,.1];
mu_true = [-10 0 10 0;0 0 0 10]';
Sigma_true(:,:,1) = [3 1;1 20];
Sigma_true(:,:,2) = [7 1;1 2];
Sigma_true(:,:,3) = [4 1;1 16];
Sigma_true(:,:,4) = [4 1;1 16];

%Creat GMM
gm = gmdistribution(mu_true, Sigma_true,alpha_true);
rng('default');
N = 1000;
[Y, compIdx] = random(gm,N);

% True GMM visualization
scatter(Y(:,1),Y(:,2),3,compIdx);
title('GMM');
xlabel('Feature 1')
ylabel('FEature 2')

%% Generate Samples
N = [10,100,1000,10000];
datasets = {};
for i=1:4
    %subplot
    [Y, compIdx] = random(gm,N(i));
    datasets{i} = {Y,compIdx};
    %scatter(Y(:,1),Y(:,2),3,compIdx);
end

%% Cross Validation 
figure(2)
for i=1:4
    subplot(2,2,i)
    data = datasets{i}{1};
    labels = datasets{i}{2};
    indices = crossvalind('Kfold',labels,10);
    fprintf("Dataset %d\n",i)
    Model_Scores = [];
    for j=1:6
        for k=1:10
            test = (indices == i);
            train = ~test;
            % Iteravtive EM
            GMModel = fitgmdist(data(train),j,'RegularizationValue',0.1,'Options',statset('MaxIter',1000));
            BIC_scores(k) = GMModel.BIC;
        end
        Model_Scores = [Model_Scores mean(BIC_scores)];
    end
    plot(1:length(Model_Scores),Model_Scores,'linewidth',3);
    title("Number of Data Samples " + num2str(size(data,1)))
    ylabel("BIC Score")
    xlabel("Number of GMM Components")
    fprintf("Order %d, BIC Score %2.4f\n",[(1:6)' Model_Scores']')
    [~,idx] = min(Model_Scores);
    fprintf("Best order of GMM %d\n",idx);
    fprintf("\n")
end




