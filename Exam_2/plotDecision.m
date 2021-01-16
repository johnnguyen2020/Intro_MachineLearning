function plotDecision(data,ind01,ind10,ind00,ind11)

plot(data(ind01,1),data(ind01,2),'xm'); hold on;  % false negatives
plot(data(ind10,1),data(ind10,2),'om'); hold on;  % false positives
plot(data(ind00,1),data(ind00,2),'xg'); hold on;
plot(data(ind11,1),data(ind11,2),'og'); hold on;

xlabel('Feature 1', 'FontSize', 16);
ylabel('Feature 2', 'FontSize', 16);
grid on; box on;
set(gca, 'FontSize', 14);
legend({'Misclassified as C1','Misclassified as C2'});

