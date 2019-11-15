%% Generate Data
close all; clear all; clc;
m(:,1) = [-1;0]; Sigma(:,:,1) = 0.1*[10 -4;-4,5]; % mean and covariance of data pdf conditioned on label 3
m(:,2) = [1;0]; Sigma(:,:,2) = 0.1*[5 0;0,2]; % mean and covariance of data pdf conditioned on label 2
m(:,3) = [0;1]; Sigma(:,:,3) = 0.1*eye(2); % mean and covariance of data pdf conditioned on label 1
classPriors = [0.15,0.35,0.5]; thr = [0,cumsum(classPriors)];
N = 10000; u = rand(1,N); L = zeros(1,N); x = zeros(2,N);
figure(1),clf, colorList = 'rbg';
for l = 1:3
    indices = find(thr(l)<=u & u<thr(l+1)); % if u happens to be precisely 1, that sample will get omitted - needs to be fixed
    L(1,indices) = l*ones(1,length(indices));
    x(:,indices) = mvnrnd(m(:,l),Sigma(:,:,l),length(indices))';
    figure(1), plot(x(1,indices),x(2,indices),'.','MarkerFaceColor',colorList(l)); axis equal, hold on,
end

legend('Dist. 1', 'Dist. 2', 'Dist. 3');
%%

% L is the label for each class in the generated data
% decision rule is decide class j if p(c_k | x) > p(c_j | x) for all j
pc1_x = mvnpdf(x', m(:,1)', Sigma(:,:,1)) * classPriors(1);
pc2_x = mvnpdf(x', m(:,2)', Sigma(:,:,2)) * classPriors(2);
pc3_x = mvnpdf(x', m(:,3)', Sigma(:,:,3)) * classPriors(3);
[~,predictions] = max([pc1_x,pc2_x,pc3_x],[],2);
%% a
tabulate(L)
%% b
C = confusionmat(L',predictions)
figure(2)
confusionchart(C)
%% c
sum(L'~=predictions)
%% d
sum(L'~=predictions)/length(x)
%% e
figure(3)
plotClassErrors(x',L',3,predictions)