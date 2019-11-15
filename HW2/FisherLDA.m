function FisherLDA(data,classInd,threshold,tit)

if nargin < 3
   threshold = 0 
end

n = 2; 

x1 = data(classInd==1,:)';
x2 = data(classInd==2,:)';

N1 = length(x1);
N2 = length(x2);

mu1hat = mean(x1,2); S1hat = cov(x1');
mu2hat = mean(x2,2); S2hat = cov(x2');

Sb = (mu1hat-mu2hat)*(mu1hat-mu2hat)';
Sw = S1hat + S2hat;

[V,D] = eig(inv(Sw)*Sb);
[~,ind] = sort(diag(D),'descend');
w = V(:,ind(1)); % Fisher LDA projection vector

y1 = w'*x1;
y2 = w'*x2;

figure,

subplot(2,1,1), 
plot(x1(1,:),x1(2,:),'r*');
hold on;
plot(x2(1,:),x2(2,:),'bo');
axis equal,
title(tit)

subplot(2,1,2), 
plot(y1(1,:),zeros(1,N1),'r*');
hold on;
plot(y2(1,:),zeros(1,N2),'bo');
axis equal,

% find threshold muahah
y = [y1(1,:) y2(1,:)];
thresholds = linspace(min(y),max(y),10000);
optimal_threshold = 0;
optimal_accuracy = 0;

for threshold=thresholds
    
    if mean(y1) < mean(y2)
        inferred_classes = double(y < threshold)';
        inferred_classes(inferred_classes==0) = 2;
    else
        inferred_classes = double(y > threshold)';
        inferred_classes(inferred_classes==0) = 2;
    end
  
    accuracy = (sum(inferred_classes == classInd))/length(y);
    
    if accuracy > optimal_accuracy
        optimal_threshold = threshold;
        optimal_accuracy = accuracy;
    end
end
plot([optimal_threshold,optimal_threshold],[-.5,.5],'g-','linewidth',3)
title("Accuracy = "+num2str(optimal_accuracy)+" Optimal Threshold = "+num2str(optimal_threshold))
