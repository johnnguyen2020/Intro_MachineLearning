function plotSamples(data, classIndex, nClass, titleString)
markerStrings = ['x','o']; colorString = ['r', 'b'];

for idxClass = 1:nClass
    dataClass = data(classIndex==idxClass,:);
    plot(dataClass(:,1), dataClass(:,2) , [colorString(idxClass) markerStrings(idxClass)]);
    hold on;
end

hold off;
title(titleString, 'FontSize', 18);
xlabel('Feature 1', 'FontSize', 16);
ylabel('Feature 2', 'FontSize', 16);
legend({'Class 1', 'Class 2'});
grid on; box on;
set(gca, 'FontSize', 14);