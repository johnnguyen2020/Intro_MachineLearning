%% EECE5644 - Homework 2 - Question 2 and 3
clear all; close all; clc;
rng('default');

%% Dataset 1
nSamples = 999;
mu{1} = [0,0];
sigma{1} = [2 0.5; 0.5 1];
mu{2} = [3,3];
sigma{2} = [2 -1.9; -1.9 5];
prior = [0.3; 0.7];

xMin = -5;
xMax = 10;
yMin = -5;
yMax = 12;

nClass = numel(mu);
[data, classIndex] = generateGaussianSamples(mu, sigma, nSamples, prior);
figure(1); 
%subplot(1,3,1);
titleString = 'Samples';
plotSamples(data, classIndex, nClass, titleString);
axis([xMin xMax yMin yMax]);

% MAP Classification and Visualization for Question 2
[ind01MAP,ind10MAP,ind00MAP,ind11MAP,pEminERM] = classifyMAP(data, classIndex, mu, sigma, nSamples, prior);
figure(2); 
%subplot(1,3,2);
plotDecision(data,ind01MAP,ind10MAP,ind00MAP,ind11MAP);
title(sprintf('MAP Pe=%.4f',pEminERM), 'FontSize', 18);
axis([xMin xMax yMin yMax]);

% LDA Classification and Visualization for Question 3
[ind01LDA,ind10LDA,ind00LDA,ind11LDA,pEminLDA] = classifyLDA(data, classIndex, mu, sigma, nSamples, prior);
figure(3); 
%subplot(1,3,3);
plotDecision(data,ind01LDA,ind10LDA,ind00LDA,ind11LDA);
title(sprintf('LDA Pe=%.4f',pEminLDA), 'FontSize', 18);
axis([xMin xMax yMin yMax]);

FisherLDA(data,classIndex,0,'Fisher LDA')

%% LOGISTIC REGRESSION
% x is column vector
% w is column vector
% X is nxm matrix of data samples
w_old = randn(3,1);
X = [data ones(nSamples,1)];
t = double(~(classIndex-1));

% Newton Rhapson iterative optimization 
for i = 1:100
y = @(x) 1./(1+exp(-(w_old'*x)));
Y = y(X')'; %probability predications for all classes
R = diag(Y.*(1-Y));
z = X*w_old - inv(R + eye(nSamples)*eps) *(Y-t);
w_new = inv(X'*R*X+eye(3,3)*eps) * X' * R * z;
if norm(w_new-w_old,2)<1e-10
    break
end
w_old = w_new;
end

sum(((Y>.5)~=t))
logistic_classification = uint8((Y>.5))
logistic_classification(logistic_classification==0) = 2;

xMin = min(X(:,1));
xMax = max(X(:,1));
yMin = min(X(:,2));
yMax = max(X(:,2));

[X,Y] = meshgrid(linspace(xMin,xMax,100),linspace(yMin,yMax,100));
data_MESH = [X(:) Y(:) ones(100*100,1)];
decision_boundary = y(data_MESH');
max_d = max(decision_boundary);
min_d = min(decision_boundary);

% LOGISTIC 
figure(5)
hold on
plotClassErrors(data, classIndex,nClass, logistic_classification, 'LOGISTIC REGRESSION');
contour(X,Y,reshape(decision_boundary,100,100),[min_d*[.49],5,[0.51]*max_d])
axis([xMin xMax yMin yMax]);



