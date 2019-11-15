close all, clear all, clc,
N=1000; 
n = 2; 
K=10;
p = [0.35,0.65]; % class priors for labels 0 and 1 respectively

% Generate training samples
label = rand(1,N) >= p(1); 
l = 2*(label-0.5);
Nc = [length(find(label==0)),length(find(label==1))]; % number of samples from each class
x = zeros(n,N); % reserve space

% Draw samples from each class pdf
for lbl = 0:1
    if lbl==0
        x(:,label==lbl) = randGaussian(Nc(lbl+1),[0;0],eye(2));
    elseif lbl==1
        r = 2 + rand(Nc(lbl+1),1);
        theta = -pi + 2* pi .*rand(Nc(lbl+1),1);
        x(:,label==lbl) = [r.*cos(theta) r.*sin(theta)]';
    end
end

%Generate independent test samples
label2 = rand(1,N) >= p(1); 
l2 = 2*(label2-0.5);
Nc2 = [length(find(label2==0)),length(find(label2==1))]; % number of samples from each class
x2 = zeros(n,N); % reserve space

% Draw samples from each class pdf
for lbl2 = 0:1
    if lbl2==0
        x2(:,label2==lbl2) = randGaussian(Nc2(lbl2+1),[0;0],eye(2));
    elseif lbl2==1
        r2 = 2 + rand(Nc2(lbl2+1),1);
        theta2 = -pi + 2* pi .*rand(Nc2(lbl2+1),1);
        x2(:,label2==lbl2) = [r2.*cos(theta2) r2.*sin(theta2)]';
    end
end

figure(3)
scatter(x(1,:),x(2,:),25,l+1);
title('Training Data Samples')
xlabel('x1'), ylabel('x2'), xlim([-5 5]), ylim([-5 5])


%%
% Train a Linear kernel SVM with cross-validation
% to select hyperparameters that minimize probability 
% of error (i.e. maximize accuracy; 0-1 loss scenario)
dummy = ceil(linspace(0,N,K+1));
for k = 1:K, indPartitionLimits(k,:) = [dummy(k)+1,dummy(k+1)]; end,
CList = 10.^linspace(-3,7,11);
for CCounter = 1:length(CList)
    [CCounter,length(CList)];
    C = CList(CCounter);
    for k = 1:K
        indValidate = [indPartitionLimits(k,1):indPartitionLimits(k,2)];
        xValidate = x(:,indValidate); % Using folk k as validation set
        lValidate = l(indValidate);
        if k == 1
            indTrain = [indPartitionLimits(k,2)+1:N];
        elseif k == K
            indTrain = [1:indPartitionLimits(k,1)-1];
        else
            indTrain = [indPartitionLimits(k-1,2)+1:indPartitionLimits(k+1,1)-1];
        end
        % using all other folds as training set
        xTrain = x(:,indTrain); lTrain = l(indTrain);
        SVMk = fitcsvm(xTrain',lTrain,'BoxConstraint',C,'KernelFunction','linear');
        dValidate = SVMk.predict(xValidate')'; % Labels of validation data using the trained SVM
        indCORRECT = find(lValidate.*dValidate == 1); 
        Ncorrect(k)=length(indCORRECT);
    end 
    PCorrect(CCounter)= sum(Ncorrect)/N; 
end 
figure(1), subplot(1,3,1),
plot(log10(CList),PCorrect,'.',log10(CList),PCorrect,'-'),
[idgaf,idgaf2] = max(PCorrect);
fprintf("Minimum CV Probability of Error %f @ c=%f\n",1-idgaf,CList(idgaf2));
xlabel('log_{10} C'),ylabel('K-fold Validation Accuracy Estimate'),
title('Linear-SVM Cross-Val Accuracy Estimate'), %axis equal,
[dummy,indi] = max(PCorrect(:)); [indBestC, indBestSigma] = ind2sub(size(PCorrect),indi);
CBest= CList(indBestC); 
SVMBest = fitcsvm(x',l','BoxConstraint',CBest,'KernelFunction','linear');
d = SVMBest.predict(x')'; % Labels of training data using the trained SVM
indINCORRECT = find(l.*d == -1); % Find training samples that are incorrectly classified by the trained SVM
indCORRECT = find(l.*d == 1); % Find training samples that are correctly classified by the trained SVM
figure(1), subplot(1,3,2), 
plot(x(1,indCORRECT),x(2,indCORRECT),'g.'), hold on,
plot(x(1,indINCORRECT),x(2,indINCORRECT),'r.'), axis equal,
title('Training Data (RED: Incorrectly Classified)'),
pTrainingError = length(indINCORRECT)/N;% Empirical estimate of training error probability
fprintf("Number incorrectly classified : %d\nPercentage of Error %f\n",length(indINCORRECT),pTrainingError) 
Nx = 1001; Ny = 990; xGrid = linspace(-10,10,Nx); yGrid = linspace(-10,10,Ny);
[h,v] = meshgrid(xGrid,yGrid); dGrid = SVMBest.predict([h(:),v(:)]); zGrid = reshape(dGrid,Ny,Nx);
figure(1), subplot(1,3,2), contour(xGrid,yGrid,zGrid,0); xlabel('x1'), ylabel('x2'), axis equal,

d = SVMBest.predict(x2')';
indINCORRECT = find(l2.*d == -1); % Find training samples that are incorrectly classified by the trained SVM
indCORRECT = find(l2.*d == 1); % Find training samples that are correctly classified by the trained SVM
figure(1), subplot(1,3,3), 
plot(x2(1,indCORRECT),x2(2,indCORRECT),'g.'), hold on,
plot(x2(1,indINCORRECT),x2(2,indINCORRECT),'r.'), axis equal,
title('Testing Data (RED: Incorrectly Classified)'),
pTrainingError = length(indINCORRECT)/N;% Empirical estimate of training error probability
fprintf("Number incorrectly classified : %d\nPercentage of Error %f\n",length(indINCORRECT),pTrainingError) 
Nx = 1001; Ny = 990; xGrid = linspace(-10,10,Nx); yGrid = linspace(-10,10,Ny);
[h,v] = meshgrid(xGrid,yGrid); dGrid = SVMBest.predict([h(:),v(:)]); zGrid = reshape(dGrid,Ny,Nx);
figure(1), subplot(1,3,3), contour(xGrid,yGrid,zGrid,0); xlabel('x1'), ylabel('x2'), axis equal,


%% Train a Gaussian kernel SVM with cross-validation
% to select hyperparameters that minimize probability 
% of error (i.e. maximize accuracy; 0-1 loss scenario)
dummy = ceil(linspace(0,N,K+1));
for k = 1:K, indPartitionLimits(k,:) = [dummy(k)+1,dummy(k+1)]; end,
CList = 10.^linspace(-1,9,11); sigmaList = 10.^linspace(-2,3,13);
for sigmaCounter = 1:length(sigmaList)
    [sigmaCounter,length(sigmaList)];
    sigma = sigmaList(sigmaCounter);
    for CCounter = 1:length(CList)
        C = CList(CCounter);
        for k = 1:K
            indValidate = [indPartitionLimits(k,1):indPartitionLimits(k,2)];
            xValidate = x(:,indValidate); % Using folk k as validation set
            lValidate = l(indValidate);
            if k == 1
                indTrain = [indPartitionLimits(k,2)+1:N];
            elseif k == K
                indTrain = [1:indPartitionLimits(k,1)-1];
            else
                indTrain = [indPartitionLimits(k-1,2)+1:indPartitionLimits(k+1,1)-1];
            end
            % using all other folds as training set
            xTrain = x(:,indTrain); lTrain = l(indTrain);
            SVMk = fitcsvm(xTrain',lTrain,'BoxConstraint',C,'KernelFunction','gaussian','KernelScale',sigma);
            dValidate = SVMk.predict(xValidate')'; % Labels of validation data using the trained SVM
            indCORRECT = find(lValidate.*dValidate == 1); 
            Ncorrect(k)=length(indCORRECT);
        end 
        PCorrect(CCounter,sigmaCounter)= sum(Ncorrect)/N;
    end 
end
figure(2), subplot(1,3,1),
contour(log10(CList),log10(sigmaList),PCorrect',20); xlabel('log_{10} C'), ylabel('log_{10} sigma'),
[max_p,idx_] = max(PCorrect,[],[1 2],'linear');
[r,c] = ind2sub(size(PCorrect),idx_);
fprintf("Minimum CV Probability of Error %f @ c=%f sigma=%f\n",1-max_p,CList(r),sigmaList(c));
title('Gaussian-SVM Cross-Val Accuracy Estimate'), axis equal,
[dummy,indi] = max(PCorrect(:)); [indBestC, indBestSigma] = ind2sub(size(PCorrect),indi);
CBest= CList(indBestC); sigmaBest= sigmaList(indBestSigma); 
SVMBest = fitcsvm(x',l','BoxConstraint',CBest,'KernelFunction','gaussian','KernelScale',sigmaBest);
d = SVMBest.predict(x')'; % Labels of training data using the trained SVM
indINCORRECT = find(l.*d == -1); % Find training samples that are incorrectly classified by the trained SVM
indCORRECT = find(l.*d == 1); % Find training samples that are correctly classified by the trained SVM
figure(2), subplot(1,3,2), 
plot(x(1,indCORRECT),x(2,indCORRECT),'g.'), hold on,
plot(x(1,indINCORRECT),x(2,indINCORRECT),'r.'), axis equal,
title('Training Data (RED: Incorrectly Classified)'),
pTrainingError = length(indINCORRECT)/N; % Empirical estimate of training error probability
fprintf("Number incorrectly classified : %d\nPercentage of Error %f\n",length(indINCORRECT),pTrainingError) 
Nx = 1001; Ny = 990; xGrid = linspace(-10,10,Nx); yGrid = linspace(-10,10,Ny);
[h,v] = meshgrid(xGrid,yGrid); dGrid = SVMBest.predict([h(:),v(:)]); zGrid = reshape(dGrid,Ny,Nx);
figure(2), subplot(1,3,2), contour(xGrid,yGrid,zGrid,0); xlabel('x1'), ylabel('x2'), axis equal,

d = SVMBest.predict(x2')';
indINCORRECT = find(l2.*d == -1); % Find training samples that are incorrectly classified by the trained SVM
indCORRECT = find(l2.*d == 1); % Find training samples that are correctly classified by the trained SVM
figure(2), subplot(1,3,3), 
plot(x2(1,indCORRECT),x2(2,indCORRECT),'g.'), hold on,
plot(x2(1,indINCORRECT),x2(2,indINCORRECT),'r.'), axis equal,
title('Testing Data (RED: Incorrectly Classified)'),
pTrainingError = length(indINCORRECT)/N;% Empirical estimate of training error probability
fprintf("Number incorrectly classified : %d\nPercentage of Error %f\n",length(indINCORRECT),pTrainingError) 
Nx = 1001; Ny = 990; xGrid = linspace(-10,10,Nx); yGrid = linspace(-10,10,Ny);
[h,v] = meshgrid(xGrid,yGrid); dGrid = SVMBest.predict([h(:),v(:)]); zGrid = reshape(dGrid,Ny,Nx);
figure(2), subplot(1,3,3), contour(xGrid,yGrid,zGrid,0); xlabel('x1'), ylabel('x2'), axis equal,













