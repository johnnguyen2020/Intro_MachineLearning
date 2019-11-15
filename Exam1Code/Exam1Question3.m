clear all; clc; close all;
%% define parameters
N = 10;
sigma=.01;
w_true = [.5;-1;-.8;1];
%% create dataset
v = mvnrnd(0,sigma,N);
%x = linspace(-3,3,N)';
x = sort(unifrnd(-1,1,[N,1]));
phi_x = [x.^3 x.^2 x ones(N,1)]

y_true = phi_x *w_true;
y_corrupted = y_true + v
%% choose gamma
B = .01;
gamma_values = linspace(.07,10^B,10);
for gamma = gamma_values
close all;
figure(1)
hold on;
plot(x,y_corrupted,'b-')
plot(x,y_true,'r-','linewidth',1)
%% 
w_map = ((y_corrupted'*phi_x) / ((phi_x'*phi_x)/(sigma^2) - N*gamma^(-2)*eye(length(w_true))))' * 1/(sigma^2) 
y_estimated = phi_x*w_map
plot(x,y_estimated,'g-','linewidth',1)
legend(["Corrupted Signal","True Signal","MAP Estimate"]) 
title("Samples = "+num2str(N)+", Gamma = " + num2str(gamma))
saveas(gcf,"Gamma="+num2str(gamma)+".jpg")
pause
close all;                    
end