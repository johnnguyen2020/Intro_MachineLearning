function net = define_NN(activation_function,neurons)
    net = feedforwardnet(neurons);
    net.layers{1}.transferFcn = activation_function;
    net.divideParam.trainRatio=1.0;
    net.divideParam.valRatio=0;
    net.divideParam.testRatio=0;
    net.trainParam.showWindow=0;
    net.trainParam.epochs=2500;
end