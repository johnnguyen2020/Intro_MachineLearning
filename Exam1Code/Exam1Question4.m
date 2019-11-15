%% Generate random point
close all;
clear all;
%% pick position in unit circle
radius = .6 + .4*rand(); % random radius
theta = 0 + (2*pi)*rand(); % random theta in radians
position = [radius * cos(theta) , radius * sin(theta)] % random vehicle position inside the unit circle
%% generate evenly spaced K landmarks
%% optimization
sigmax = .3;
sigmay = sigmax;
K = 4
for k = 1:K
theta_k = linspace(-pi,pi,k+1);
theta_k = theta_k(1:end-1);
reference_positions = [cos(theta_k)' , sin(theta_k)'];
%% sanity check...
subplot(2,2,k)
set(gca, 'YAxisLocation', 'origin')
plot(reference_positions(:,1),reference_positions(:,2),'r*') % check if evenly spaced
hold on
plot(position(1),position(2),'b*')
title(num2str(k) + " reference points")
xlabel("X")
ylabel("Y")
axis equal
%% generate K range measurements according to model
sigma_measurement = .3;
for i = 1:k
r(i) = pdist([position;reference_positions(i,:)],'euclidean') + normrnd(0,sigma_measurement);
end
horizontalGrid = linspace(-2,2,101);
verticalGrid = linspace(-2,2,101);
[h,v] = meshgrid(horizontalGrid,verticalGrid);
hh = h(:);
vv = v(:);
for i = 1:length(hh)
params = [hh(i),vv(i)];
obj_fun(i) = sum(((r-sqrt(sum((repmat(params,k,1) - reference_positions).^2,2))').^2)/(sigma_measurement^2)) + params * inv([sigmax^2 0; 0 sigmay^2]) * params';
end
minDSGV = min(obj_fun);
maxDSGV = max(obj_fun);
discriminantScoreGrid = reshape(obj_fun,101,101);
contour(horizontalGrid,verticalGrid,discriminantScoreGrid,10.^linspace(1.5,2.5,5));
end