clear all; clc; close all;
%% define parameters
N = 10;
sigma=.01;
w_true = [.5;-1;-.8;1];
%% select gamma values and initialize arrays to fill
B = .01;
gamma_values = linspace(.07,10^B,100);
maxs=[];
mins=[];
lower_quart = [];
median_ = [];
upper_quart = [];
for gamma = gamma_values
    l2_norms = [];
        for i = 1:100
            %% create dataset
            v = mvnrnd(0,sigma,N);
            x = sort(unifrnd(-1,1,[N,1]));
            % phi_x -> training example is each row
            phi_x = [x.^3 x.^2 x ones(N,1)];
            y_true = phi_x *w_true;
            y_corrupted = y_true + v;
            w_map = ((y_corrupted'*phi_x) / ((phi_x'*phi_x)/(sigma^2) - N*gamma^(-2)*eye(length(w_true))))' * 1/(sigma^2) ;
            y_estimated = phi_x*w_map;
            l2_norms = [l2_norms norm(w_true-w_map,2)^2];
        end
    maxs=[maxs max(l2_norms)];
    mins=[mins min(l2_norms)];
    lower_quart = [lower_quart quantile(l2_norms,.25)];
    median_ = [median_ median(l2_norms)];
    upper_quart = [upper_quart quantile(l2_norms,.75)];
end
%%
figure(1)

subplot(2,3,1)
stem(gamma_values,maxs,'r-')
title('Maximum')
xlabel("Gamma")
ylabel("||w_{true}-w_{MAP}||")

subplot(2,3,2)
stem(gamma_values,median_,'b-')
title('Median')
xlabel("Gamma")
ylabel("||w_{true}-w_{MAP}||")

subplot(2,3,3)
stem(gamma_values,mins,'g-')
title('Minimum')
xlabel("Gamma")
ylabel("||w_{true}-w_{MAP}||")

subplot(2,3,4)
stem(gamma_values,lower_quart)
title("25%")
xlabel("Gamma")
ylabel("||w_{true}-w_{MAP}||")

subplot(2,3,5)
stem(gamma_values,upper_quart)
title("75%")
xlabel("Gamma")
ylabel("||w_{true}-w_{MAP}||")