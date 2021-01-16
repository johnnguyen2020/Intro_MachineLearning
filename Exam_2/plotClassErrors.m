function plotClassErrors(data, classIndex,nClass,my_inference, tit)
markerStrings = 'xo.';
errors = my_inference~=classIndex;
legendStrings = [];
for idxClass = 1:nClass
    idx_correct = (~errors) & (classIndex==idxClass);
    dataClass = data(idx_correct,:);
    plot(dataClass(:,1), dataClass(:,2) , ['g' markerStrings(idxClass)]);
    hold on
    if ~isempty(dataClass)
        legendStrings = [legendStrings "Class "+num2str(idxClass)+" Correct"];
    end
    
    idx_incorrect = (errors) & (classIndex==idxClass);
    dataClass = data(idx_incorrect,:);
    plot(dataClass(:,1), dataClass(:,2) , ['r' markerStrings(idxClass)]);
    hold on
    if ~isempty(dataClass)
        legendStrings = [legendStrings "Class "+num2str(idxClass)+" Wrong"];
    end
    
    %axis image
    axis([-5 10 -5 12]);
end
xlabel('Feature 1');
ylabel('Feature 2');

legend(legendStrings);
%title("Accuracy = "+num2str(1-sum(errors)/length(data))+ " | P(error) = "+num2str(1-(1-sum(errors)/length(data)))) 
title(tit + " Pe = "+num2str(1-(1-sum(errors)/length(data)))) 
axis([-5 10 -5 12]);
grid on; box on;
set(gca, 'FontSize', 14);