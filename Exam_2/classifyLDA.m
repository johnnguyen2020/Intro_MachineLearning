function [ind01LDA,ind10LDA,ind00LDA,ind11LDA,pEminLDA] = classifyLDA(data, classIndex, mu, sigma, nSamples, prior)

% Fisher LDA Classifer (using true model parameters)
Sb = (mu{1}'-mu{2}')*(mu{1}'-mu{2}')';
Sw = sigma{1} + sigma{2};
[V,D] = eig(inv(Sw)*Sb); % LDA solution satisfies alpha Sw w = Sb w; ie w is a generalized eigenvector of (Sw,Sb)
[~,ind] = sort(diag(D),'descend');
wLDA = V(:,ind(1)); % Fisher LDA projection vector
yLDA = wLDA'*data'; % All data projected on to the line spanned by wLDA
wLDA = sign(mean(yLDA(find(classIndex==2)))-mean(yLDA(find(classIndex==1))))*wLDA; % ensures class1 falls on the + side of the axis
discriminantScoreLDA = sign(mean(yLDA(find(classIndex==2)))-mean(yLDA(find(classIndex==1))))*yLDA; % flip yLDA accordingly

% Estimate the ROC curve for this LDA classifier
[ROCLDA,tauLDA] = estimateROC(discriminantScoreLDA,classIndex');
probErrorLDA = [ROCLDA(1,:)',1-ROCLDA(2,:)']*[sum(classIndex==1),sum(classIndex==2)]'/nSamples; % probability of error for LDA for different threshold values
pEminLDA = min(probErrorLDA);   % minimum probability of error

ind = find(probErrorLDA == pEminLDA);
decisionLDA = (discriminantScoreLDA >= tauLDA(ind(1))); % use smallest min-error threshold
ind00LDA = find(decisionLDA==0 & classIndex'==1); % true negatives
ind10LDA = find(decisionLDA==1 & classIndex'==1); % false positives
ind01LDA = find(decisionLDA==0 & classIndex'==2); % false negatives
ind11LDA = find(decisionLDA==1 & classIndex'==2); % true positives

function [ROC,tau] = estimateROC(discriminantScoreLDA,label)
% Generate ROC curve samples
Nc = [length(find(label==1)),length(find(label==2))];
sortedScore = sort(discriminantScoreLDA,'ascend');
tau = [sortedScore(1)-1,(sortedScore(2:end)+sortedScore(1:end-1))/2,sortedScore(end)+1];
% thresholds at midpoints of consecutive scores in sorted list
for k = 1:length(tau)
    decision = (discriminantScoreLDA >= tau(k));
    ind10 = find(decision==1 & label==1); p10 = length(ind10)/Nc(1); % probability of false positive
    ind11 = find(decision==1 & label==2); p11 = length(ind11)/Nc(2); % probability of true positive
    ROC(:,k) = [p10;p11];
end

