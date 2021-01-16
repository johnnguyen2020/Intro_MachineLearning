function net = setupNN(activation_function,neurons)
    net = patternnet(neurons);
%     net.layers{1}.transferFcn = activation_function;
%     net.layers{2}.transferFcn = "softmax";
    net.divideParam.trainRatio=0.80;
    net.divideParam.valRatio=0.20;
    net.divideParam.testRatio=0;
    net.trainParam.showWindow=0;
    net.trainParam.epochs=2500;
%     net.performFcn = "crossentropy";
end