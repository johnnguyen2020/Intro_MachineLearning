%% Init
clear all; close all; clc; 

labels = {'Feature 1', 'Feature 2'};
legendKey = {'Class 1', 'Class 2'};

markers = ['x','o'];
colors = ['r', 'b'];
%% 2a
n_samples = 400;
mu{1} = [0,0];
sigma{1} = eye(2);
mu{2} = [3,3];
sigma{2} = eye(2);
prior = [0.5; 0.5];

xMin = -3;
xMax = 6;
yMin = -3;
yMax = 6;

titles = '2a';

nClass = numel(mu);

[data, classIndex] = generateSamples(n_samples, prior, mu, sigma);

figure(1)
subplot(2,3,1)
plotSamples(data, classIndex, nClass, titles, labels, legendKey, colors, markers);
axis([xMin xMax yMin yMax]);
plotDecisionBoundary(xMin,xMax,yMin,yMax,mu,sigma,prior);



figure(2)
subplot(2,3,1)
my_inference = inferClassLabel(data,mu,sigma,prior);
plotClassErrors(data, classIndex,nClass,my_inference);
axis([xMin xMax yMin yMax]);
plotDecisionBoundary(xMin,xMax,yMin,yMax,mu,sigma,prior)

FisherLDA(data,classIndex,0,titles)
%% 2b
n_samples = 400;
mu{1} = [0,0];
sigma{1} = [3, 1; 1, 0.8];
mu{2} = [3,3];
sigma{2} = [3, 1; 1, 0.8];
prior = [0.5; 0.5];
titles = '2b';

xMin = -5;
xMax = 7;
yMin = -3;
yMax = 6;

nClass = numel(mu);
[data, classIndex] = generateSamples(n_samples, prior, mu, sigma);

figure(1)
subplot(2,3,2)
plotSamples(data, classIndex, nClass, titles, labels, legendKey, colors, markers);
axis([xMin xMax yMin yMax])
plotDecisionBoundary(xMin,xMax,yMin,yMax,mu,sigma,prior);

figure(2)
subplot(2,3,2)
my_inference = inferClassLabel(data,mu,sigma,prior);
plotClassErrors(data, classIndex,nClass,my_inference);
axis([xMin xMax yMin yMax]);
plotDecisionBoundary(xMin,xMax,yMin,yMax,mu,sigma,prior)

FisherLDA(data,classIndex,0,titles)
%% 2c
n_samples = 400;
mu{1} = [0,0];
sigma{1} = [2 0.5; 0.5 1];
mu{2} = [2,2];
sigma{2} = [2 -1.9; -1.9 5];
prior = [0.5; 0.5];
titles = '2c';

xMin = -5;
xMax = 7;
yMin = -4;
yMax = 7;

nClass = numel(mu);
[data, classIndex] = generateSamples(n_samples, prior, mu, sigma);

figure(1)
subplot(2,3,3)
plotSamples(data, classIndex, nClass, titles, labels, legendKey, colors, markers);
axis([xMin xMax yMin yMax])
plotDecisionBoundary(xMin,xMax,yMin,yMax,mu,sigma,prior);

figure(2)
subplot(2,3,3)
my_inference = inferClassLabel(data,mu,sigma,prior);
plotClassErrors(data, classIndex,nClass,my_inference);
axis([xMin xMax yMin yMax]);
plotDecisionBoundary(xMin,xMax,yMin,yMax,mu,sigma,prior)

FisherLDA(data,classIndex,0,titles)
%% 2d

% P7d
n_samples = 400;
mu{1} = [0,0];
sigma{1} = eye(2);
mu{2} = [3,3];
sigma{2} = eye(2);
prior = [0.05; 0.95];

xMin = -3;
xMax = 6;
yMin = -3;
yMax = 6;

titles = '2d';

nClass = numel(mu);
[data, classIndex] = generateSamples(n_samples, prior, mu, sigma);

figure(1)
subplot(2,3,4)
plotSamples(data, classIndex, nClass, titles, labels, legendKey, colors, markers);
axis([xMin xMax yMin yMax]);
plotDecisionBoundary(xMin,xMax,yMin,yMax,mu,sigma,prior);

figure(2)
subplot(2,3,4)
my_inference = inferClassLabel(data,mu,sigma,prior);
plotClassErrors(data, classIndex,nClass,my_inference);
axis([xMin xMax yMin yMax]);
plotDecisionBoundary(xMin,xMax,yMin,yMax,mu,sigma,prior)

FisherLDA(data,classIndex,0,titles)

%% 2e
n_samples = 400;
mu{1} = [0,0];
sigma{1} = [3, 1; 1, 0.8];
mu{2} = [3,3];
sigma{2} = [3, 1; 1, 0.8];
prior = [0.05; 0.95];
titles = '2e';

xMin = -5;
xMax = 7;
yMin = -3;
yMax = 6;

nClass = numel(mu);
[data, classIndex] = generateSamples(n_samples, prior, mu, sigma);

figure(1)
subplot(2,3,5)
plotSamples(data, classIndex, nClass, titles, labels, legendKey, colors, markers);
axis([xMin xMax yMin yMax])
plotDecisionBoundary(xMin,xMax,yMin,yMax,mu,sigma,prior);

figure(2)
subplot(2,3,5)
my_inference = inferClassLabel(data,mu,sigma,prior);
plotClassErrors(data, classIndex,nClass,my_inference);
axis([xMin xMax yMin yMax]);
plotDecisionBoundary(xMin,xMax,yMin,yMax,mu,sigma,prior)

FisherLDA(data,classIndex,0,titles)


%% 2f

% P6e
n_samples = 400;
mu{1} = [0,0];
sigma{1} = [2 0.5; 0.5 1];
mu{2} = [2,2];
sigma{2} = [2 -1.9; -1.9 5];
prior = [0.05; 0.95];
titles = '2f';

xMin = -5;
xMax = 7;
yMin = -4;
yMax = 7;

nClass = numel(mu);
[data, classIndex] = generateSamples(n_samples, prior, mu, sigma);

figure(1)
subplot(2,3,6)
plotSamples(data, classIndex, nClass, titles, labels, legendKey, colors, markers);
axis([xMin xMax yMin yMax])
plotDecisionBoundary(xMin,xMax,yMin,yMax,mu,sigma,prior)

figure(2)
subplot(2,3,6)
my_inference = inferClassLabel(data,mu,sigma,prior);
plotClassErrors(data, classIndex,nClass,my_inference);
axis([xMin xMax yMin yMax]);
plotDecisionBoundary(xMin,xMax,yMin,yMax,mu,sigma,prior)

FisherLDA(data,classIndex,0,titles)









